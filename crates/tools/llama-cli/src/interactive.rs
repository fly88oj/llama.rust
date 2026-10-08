//! interactive.rs — port of `tools/completion/completion.cpp` (pinned
//! bd4f514db1, 961 lines): the interactive REPL of the reference's
//! direct-inference CLI (`llama-completion`), grafted onto the port's
//! llama-cli as its `-i` surface.
//!
//! The pinned tree has two CLIs: `llama-cli` (an HTTP client driving
//! llama-server — tools/cli/*) and `llama-completion` (this driver). The
//! port's llama-cli is the direct-inference tool by the documented batch-1
//! architecture decision, so the interactive surface lands here and the
//! client-server CLI remains a documented gap (PARITY.md).
//!
//! Mapping (completion.cpp → Rust):
//!   `llama_completion` prologue (:86-204)   -> [`run`] setup + the port's
//!                                               own model load in main.rs
//!   session load/match (:206-356)           -> [`run`]'s session block (the
//!                                               port persists the token
//!                                               stream, not llama_state —
//!                                               documented deviation)
//!   sigint_handler (:62-81)                 -> the SIGINT hook below
//!   main loop (:545-940)                    -> [`run`]'s loop
//!   `chat_add_and_format` (:238-246)        -> [`chat_add_and_format`]
//!   `common_chat_format_single` (chat.cpp)  -> [`chat_format_single`]
//!
//! Transcript surface (what the scripted byte-for-byte parity pins, all
//! LOG()-level == stdout in the C, log.cpp:113):
//!   * the prompt echo and per-token generation pieces (`LOG("%s")`, :710)
//!   * `<<input too long: skipped N tokens>>` (:558)
//!   * the conversation `"\n> "` marker (:808)
//!   * `"EOF by user"` (:838) and `" [end of text]\n"` (:930)
//!   * the `--in-prefix`/`--in-suffix` echoes (:819/:856)
//! everything else (LOG_INF/WRN/ERR) is stderr and the port keeps its own
//! diagnostics there.

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

use llama::chat_tools::{chat_templates_apply, ChatMsg, ChatTemplates, TemplatesInputs};
use llama::console::{Console, DisplayType};
use llama::context::DecodeContext;
use llama::sampling::{GrammarSampler, SamplingContext};
use llama::vocab::{Token, Vocab, TOKEN_NULL};

use crate::Args;

/// `-no-cnv`/`-cnv` values (`common_conversation_mode`, common.h:137-141)
pub const CONVERSATION_DISABLED: u8 = 0;
pub const CONVERSATION_ENABLED: u8 = 1;
pub const CONVERSATION_AUTO: u8 = 2;

// ---------------------------------------------------------------------------
// SIGINT (:62-81) — interject in interactive mode, exit(130) on repeat
// ---------------------------------------------------------------------------

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static IS_GENERATING: AtomicBool = AtomicBool::new(false);
static INTERACTIVE_MODE: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

const SIGINT: i32 = 2;

extern "C" fn sigint_handler(_signo: i32) {
    // (:63-79): while generating, the first Ctrl+C interjects (the driver
    // checks INTERRUPTED); a second Ctrl+C exits immediately (:19-27 shape)
    if INTERRUPTED.load(Ordering::SeqCst) {
        print!("\x1b[0m\n");
        let _ = std::io::stdout().flush();
        std::process::exit(130);
    }
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// install the handler (`sigaction`, :397-403)
fn install_sigint() {
    unsafe {
        signal(SIGINT, sigint_handler as usize);
    }
}

// ---------------------------------------------------------------------------
// chat helpers — common_chat_format_single (chat.cpp) + the
// chat_add_and_format closure (completion.cpp:238-246)
// ---------------------------------------------------------------------------

/// `common_chat_format_single` (common/chat.cpp): format `past`, format
/// `past + new`, return the suffix (plus the preserved trailing newline).
fn chat_format_single(
    templates: &ChatTemplates,
    past_msg: &[ChatMsg],
    new_msg: &ChatMsg,
    add_ass: bool,
    add_bos: bool,
    add_eos: bool,
) -> String {
    let base_inputs = || TemplatesInputs {
        use_jinja: false,
        add_bos,
        add_eos,
        ..Default::default()
    };

    let mut fmt_past_msg = String::new();
    if !past_msg.is_empty() {
        let mut inputs = base_inputs();
        inputs.messages = past_msg.to_vec();
        inputs.add_generation_prompt = false;
        if let Ok(p) = chat_templates_apply(templates, &inputs) {
            fmt_past_msg = p.prompt;
        }
    }
    let mut ss = String::new();
    // if the past_msg ends with a newline, we must preserve it in the
    // formatted version
    if add_ass && fmt_past_msg.ends_with('\n') {
        ss.push('\n');
    }
    // format chat with new_msg
    let mut inputs = base_inputs();
    inputs.messages = past_msg.to_vec();
    inputs.messages.push(new_msg.clone());
    inputs.add_generation_prompt = add_ass;
    if let Ok(p) = chat_templates_apply(templates, &inputs) {
        let fmt_new_msg = p.prompt;
        ss.push_str(&fmt_new_msg[fmt_past_msg.len().min(fmt_new_msg.len())..]);
    }
    ss
}

/// `chat_add_and_format` (completion.cpp:238-246): push the message, return
/// its formatted prompt prefix.
fn chat_add_and_format(
    templates: &ChatTemplates,
    chat_msgs: &mut Vec<ChatMsg>,
    role: &str,
    content: String,
    add_bos: bool,
    add_eos: bool,
) -> String {
    let mut new_msg = ChatMsg::default();
    new_msg.role = role.to_string();
    new_msg.content = content;
    let formatted = chat_format_single(
        templates,
        chat_msgs,
        &new_msg,
        role == "user",
        add_bos,
        add_eos,
    );
    chat_msgs.push(new_msg);
    formatted
}

/// `common_token_to_piece(ctx, id, special)` (common.cpp) — the piece string.
fn piece(vocab: &Vocab, id: Token, special: bool) -> String {
    String::from_utf8_lossy(&vocab.token_to_piece_special(id, special)).into_owned()
}

/// `common_sampler_prev_str` (common/sampling.cpp:747-765): the last `n`
/// tokens' pieces, oldest first.
fn sampler_prev_str(vocab: &Vocab, sampler: &SamplingContext, n: usize) -> String {
    let n = n.min(sampler.prev.len());
    let mut result = String::new();
    for i in (0..n).rev() {
        result.push_str(&piece(vocab, sampler.prev.rat(i), false));
    }
    result
}

/// `common_sampler_last` (sampling.cpp:758-760)
fn sampler_last(sampler: &SamplingContext) -> Option<Token> {
    if sampler.prev.is_empty() {
        None
    } else {
        Some(sampler.prev.rat(0))
    }
}

/// `string_process_escapes` (common.cpp:645) — the tokenize tool's copy.
fn process_escapes(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            let replaced = match bytes[i + 1] {
                b'n' => Some(b'\n'),
                b'r' => Some(b'\r'),
                b't' => Some(b'\t'),
                b'\'' => Some(b'\''),
                b'"' => Some(b'"'),
                b'\\' => Some(b'\\'),
                _ => None,
            };
            if let Some(c) = replaced {
                out.push(c);
                i += 2;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// the driver
// ---------------------------------------------------------------------------

/// `llama_completion`'s main loop (completion.cpp:86-956). `templates` is
/// `Some` whenever a chat template is available (GGUF or `--chat-template`);
/// `has_chat_template` drives the AUTO conversation decision (:171-179).
#[allow(clippy::too_many_arguments)]
pub fn run(
    args: &Args,
    vocab: &Vocab,
    dctx: &mut DecodeContext,
    sampler: &mut SamplingContext,
    grammar: &mut Option<GrammarSampler>,
    templates: Option<ChatTemplates>,
    has_chat_template: bool,
) -> i32 {
    let n_ctx = args.n_ctx as i32;
    let n_batch = 512.min(args.n_ctx as usize) as i32;

    // conversation mode auto-enable (:170-184)
    let conversation_mode = if args.conversation_mode == CONVERSATION_AUTO {
        if has_chat_template {
            eprintln!("llama_completion: chat template is available, enabling conversation mode (disable it with -no-cnv)");
            true
        } else {
            false
        }
    } else {
        args.conversation_mode == CONVERSATION_ENABLED
    };
    if conversation_mode && !has_chat_template {
        eprintln!(
            "llama_completion: warning: chat template is not available or is not supported. \
             This may cause the model to output suboptimal responses"
        );
    }

    // `console::init(params.simple_io, params.use_color)` (:102)
    let mut console = Console::new();
    console.init(args.simple_io, args.use_color);

    // ---- the initial prompt (:248-304) ----
    let mut chat_msgs: Vec<ChatMsg> = Vec::new();
    let mut waiting_for_first_input = false;
    let mut prompt = String::new();

    if conversation_mode && args.enable_chat_template {
        let t = templates.as_ref().expect("conversation needs templates");
        if !args.system_prompt.is_empty() {
            // format the system prompt (:251-254)
            chat_add_and_format(
                t,
                &mut chat_msgs,
                "system",
                args.system_prompt.clone(),
                vocab.add_bos,
                vocab.add_eos,
            );
        }

        if !args.prompt.is_empty() {
            // format and append the user prompt (:256-259)
            chat_add_and_format(
                t,
                &mut chat_msgs,
                "user",
                args.prompt.clone(),
                vocab.add_bos,
                vocab.add_eos,
            );
        } else {
            waiting_for_first_input = true;
        }

        if !args.system_prompt.is_empty() || !args.prompt.is_empty() {
            // (:263-271)
            let inputs = TemplatesInputs {
                use_jinja: false,
                messages: chat_msgs.clone(),
                add_generation_prompt: !args.prompt.is_empty(),
                add_bos: vocab.add_bos,
                add_eos: vocab.add_eos,
                ..Default::default()
            };
            if let Ok(p) = chat_templates_apply(t, &inputs) {
                prompt = p.prompt;
            }
        }
    } else {
        // otherwise use the prompt as is (:274)
        prompt = args.prompt.clone();
    }

    let add_bos = vocab.add_bos;

    // ---- session file (:206-356; port framing: the token stream) ----
    let mut path_session = args.prompt_cache.clone();
    let mut session_tokens: Vec<Token> = Vec::new();
    if !path_session.is_empty() {
        eprintln!("llama_completion: attempting to load saved session from '{path_session}'");
        match std::fs::read(&path_session) {
            Err(_) => {
                eprintln!("llama_completion: session file does not exist, will create.");
            }
            Ok(bytes) if bytes.is_empty() => {
                eprintln!(
                    "llama_completion: The session file is empty. A new session will be initialized."
                );
            }
            Ok(bytes) => {
                // the port's session framing: 8-byte LE count + i32 LE ids.
                // The reference stores llama_state + tokens; the port cannot
                // round-trip that format, so the KV is re-evaluated on load —
                // same tokens, same matching-prefix UX, documented deviation
                if bytes.len() >= 8 {
                    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
                    if bytes.len() >= 8 + n * 4 {
                        session_tokens = bytes[8..8 + n * 4]
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                    }
                }
                if session_tokens.is_empty() {
                    eprintln!(
                        "llama_completion: warning: not a port session file, starting fresh"
                    );
                } else {
                    eprintln!(
                        "llama_completion: loaded a session with prompt size of {} tokens",
                        session_tokens.len()
                    );
                }
            }
        }
    }

    // ---- tokenize (:277-287) ----
    let mut embd_inp: Vec<Token> =
        if args.interactive_first || !prompt.is_empty() || session_tokens.is_empty() {
            vocab.tokenize(&prompt, true, true)
        } else {
            session_tokens.clone()
        };

    // Should not run without any tokens (:289-298)
    if !waiting_for_first_input && embd_inp.is_empty() {
        if add_bos {
            embd_inp.push(vocab.token_bos());
            eprintln!("llama_completion: embd_inp was considered empty and bos was added");
        } else {
            eprintln!("input is empty");
            return 1;
        }
    }

    // Tokenize negative prompt guard (:300-304)
    if embd_inp.len() as i32 > n_ctx - 4 {
        eprintln!(
            "llama_completion: prompt is too long ({} tokens, max {})",
            embd_inp.len(),
            n_ctx - 4
        );
        return 1;
    }

    // ---- session match (:308-356) ----
    let mut session_do_save;
    {
        let mut n_match = 0usize;
        if !session_tokens.is_empty() {
            for &id in &session_tokens {
                if n_match >= embd_inp.len() || id != embd_inp[n_match] {
                    break;
                }
                n_match += 1;
            }
            if args.prompt.is_empty() && n_match == embd_inp.len() {
                eprintln!("llama_completion: using full prompt from session file");
            } else if n_match >= embd_inp.len() {
                eprintln!("llama_completion: session file has exact match for prompt!");
            } else if n_match < embd_inp.len() / 2 {
                eprintln!(
                    "llama_completion: warning: session file has low similarity to prompt \
                     ({n_match} / {} tokens); will mostly be reevaluated",
                    embd_inp.len()
                );
            } else {
                eprintln!(
                    "llama_completion: session file matches {n_match} / {} tokens of prompt",
                    embd_inp.len()
                );
            }

            // remove any "future" tokens that we might have inherited from
            // the previous session (:330-341). The port re-evaluates the KV,
            // so the positions are simply dropped from the token list.
            if session_tokens.len() > n_match {
                session_tokens.truncate(n_match);
            }
        }
        session_do_save =
            !path_session.is_empty() && n_match < embd_inp.len() && !args.prompt_cache_ro;
    }

    // number of tokens to keep when resetting context (:358-363)
    let mut n_keep = args.n_keep;
    if n_keep < 0 || n_keep > embd_inp.len() as i32 {
        n_keep = embd_inp.len() as i32;
    } else {
        n_keep += add_bos as i32; // always keep the BOS token
    }

    // ---- interactive-mode resolution (:365-377) ----
    let mut interactive = args.interactive;
    let mut interactive_first = args.interactive_first;
    if conversation_mode {
        if args.single_turn && !args.prompt.is_empty() {
            interactive = false;
            interactive_first = false;
        } else {
            interactive_first = true;
        }
    }
    if interactive_first {
        interactive = true;
    }

    if interactive {
        eprintln!("llama_completion: interactive mode on.");

        for antiprompt in &args.antiprompt {
            eprintln!("Reverse prompt: '{antiprompt}'");
        }

        if args.input_prefix_bos {
            eprintln!("Input prefix with BOS");
        }
        if !args.input_prefix.is_empty() {
            eprintln!("Input prefix: '{}'", args.input_prefix);
        }
        if !args.input_suffix.is_empty() {
            eprintln!("Input suffix: '{}'", args.input_suffix);
        }
    }

    // ctrl+C handling (:396-410)
    INTERACTIVE_MODE.store(interactive, Ordering::SeqCst);
    install_sigint();

    if interactive {
        // (:474-495)
        let control_message = if args.multiline_input {
            " - To return control to the AI, end your input with '\\'.\n - To return control without starting a new line, end your input with '/'.\n"
        } else {
            " - Press Return to return control to the AI.\n - To return control without starting a new line, end your input with '/'.\n - If you want to submit another line, end your input with '\\'.\n"
        };
        eprintln!("== Running in interactive mode. ==");
        eprintln!(" - Press Ctrl+C to interject at any time.");
        eprintln!("{control_message}");
        if conversation_mode && args.enable_chat_template && args.system_prompt.is_empty() {
            eprintln!(
                " - Not using system message. To change it, set a different value via -sys PROMPT"
            );
        }
        eprintln!();
    }

    // ---- loop state (:497-525) ----
    let mut is_interacting = interactive_first;
    let mut need_insert_eot = false;
    let mut is_antiprompt = false;
    let mut input_echo = true;
    let mut display;

    let mut n_past: i32 = 0;
    let mut n_remain: i64 = args.n_predict as i64;
    let mut n_consumed: usize = 0;
    let mut n_session_consumed: usize = 0;

    let mut embd: Vec<Token> = Vec::new();

    // single-token antiprompts (:518-525)
    let antiprompt_token: Vec<Token> = args
        .antiprompt
        .iter()
        .filter_map(|ap| {
            let ids = vocab.tokenize(ap, false, true);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        })
        .collect();

    // the first thing we will do is to output the prompt, so set color
    // accordingly (:511-513)
    console.set_display(DisplayType::Prompt);
    display = args.display_prompt;

    // the logits of the last decoded token — carried across iterations the
    // way `llama_get_logits(ctx)` reads the context's last output
    let mut logits: Vec<f32> = Vec::new();
    let mut assistant_ss = String::new();

    // ---- the main loop (:545-940) ----
    loop {
        if !((n_remain != 0 && !is_antiprompt) || interactive) {
            break;
        }

        // a Ctrl+C during generation interjects (:63-68): treat it as an
        // immediate user turn
        if INTERRUPTED.load(Ordering::SeqCst) && !is_interacting {
            is_interacting = true;
            need_insert_eot = true;
            INTERRUPTED.store(false, Ordering::SeqCst);
        }

        // predict (:547-662)
        if !embd.is_empty() {
            // Note: (n_ctx - 4) here is to match the logic for commandline
            // prompt handling via --prompt or --file (:548-560)
            let max_embd_size = n_ctx - 4;
            if embd.len() as i32 > max_embd_size {
                let skipped_tokens = embd.len() as i32 - max_embd_size;
                embd.truncate(max_embd_size as usize);

                console.set_display(DisplayType::Error);
                print!(
                    "<<input too long: skipped {skipped_tokens} token{}>>",
                    if skipped_tokens != 1 { "s" } else { "" }
                );
                let _ = std::io::stdout().flush();
                console.set_display(DisplayType::Reset);
            }

            // infinite text generation via context shifting (:562-596)
            if n_past + embd.len() as i32 >= n_ctx {
                if !args.ctx_shift {
                    eprintln!(
                        "\n\nllama_completion: context full and context shift is disabled => stopping"
                    );
                    break;
                }
                if args.n_predict as i64 == -2 {
                    eprintln!(
                        "\n\nllama_completion: context full and n_predict == -2 => stopping"
                    );
                    break;
                }

                let n_left = n_past - n_keep;
                let n_discard = n_left / 2;

                eprintln!(
                    "context full, swapping: n_past = {n_past}, n_left = {n_left}, n_ctx = \
                     {n_ctx}, n_keep = {n_keep}, n_discard = {n_discard}"
                );

                dctx.kv.seq_rm(0, n_keep, n_keep + n_discard);
                dctx.kv.seq_add(0, n_keep + n_discard, n_past, -n_discard);

                n_past -= n_discard;

                // clear session path (:594-595)
                path_session.clear();
            }

            // try to reuse a matching prefix from the loaded session instead
            // of re-eval (via n_past) (:621-641). The port's session carries
            // tokens only (no state), so there is nothing to skip: the KV is
            // rebuilt by the decode below and n_past advances with it.
            if n_session_consumed < session_tokens.len() {
                let mut i = 0usize;
                while i < embd.len() {
                    if embd[i] != session_tokens[n_session_consumed] {
                        session_tokens.truncate(n_session_consumed);
                        break;
                    }
                    n_past += 1;
                    n_session_consumed += 1;
                    if n_session_consumed >= session_tokens.len() {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                if i > 0 {
                    embd.drain(..i);
                }
            }

            if !embd.is_empty() {
                let is_last_batch = n_consumed >= embd_inp.len();
                let save_now = session_do_save && is_last_batch;
                session_tokens.extend_from_slice(&embd);
                // `common_prompt_batch_decode` (common.cpp:2201-2249): the
                // embd chunks are already <= n_batch by construction
                let positions: Vec<i32> = (n_past..n_past + embd.len() as i32).collect();
                match dctx.decode(&embd, &positions) {
                    Ok(l) => logits = l.to_vec(),
                    Err(e) => {
                        eprintln!("failed to eval: {e}");
                        return 1;
                    }
                }
                n_past += embd.len() as i32;
                n_session_consumed += embd.len();
                if save_now {
                    session_do_save = false;
                }

                // Display total tokens alongside total time (:657-660)
                if args.n_print > 0 && n_past % args.n_print == 0 {
                    eprintln!("\n\nTokens consumed so far = {n_past} / {n_ctx}");
                }
            }
        }

        embd.clear();

        if embd_inp.len() <= n_consumed && !is_interacting {
            // sample the next token (:666-686)
            IS_GENERATING.store(true, Ordering::SeqCst);
            let sampled = match grammar.as_mut() {
                Some(g) => sampler.sample_with_grammar(&logits, g),
                None => Ok(sampler.sample(&logits)),
            };
            IS_GENERATING.store(false, Ordering::SeqCst);
            let id = match sampled {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("\n[grammar] {e}");
                    break;
                }
            };

            embd.push(id);

            if conversation_mode && !waiting_for_first_input && !vocab.is_eog(id) {
                assistant_ss.push_str(&piece(vocab, id, false));
            }

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            n_remain -= 1;
        } else {
            // some user input remains from prompt or interaction, forward it
            // to processing (:687-702)
            while embd_inp.len() > n_consumed {
                embd.push(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply
                // repetition penalties later; for the prompt, we don't apply
                // grammar rules
                sampler.accept(embd_inp[n_consumed]);

                n_consumed += 1;
                if embd.len() as i32 == n_batch {
                    break;
                }
            }
        }

        // display text (:704-723)
        if input_echo && display {
            for &id in &embd {
                let token_str = piece(vocab, id, args.special);

                // Console/Stream Output
                print!("{token_str}");
                let _ = std::io::stdout().flush();
            }
        }

        // reset color to default if there is no pending user input (:725-729)
        if input_echo && embd_inp.len() == n_consumed {
            console.set_display(DisplayType::Reset);
            display = true;
        }

        // if not currently processing queued inputs (:731-926)
        if embd_inp.len() <= n_consumed {
            // check for reverse prompt in the last n_prev tokens (:733-775)
            if !args.antiprompt.is_empty() {
                let n_prev = 32;
                let last_output = sampler_prev_str(vocab, sampler, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of
                // the output. If we're not running interactively, the reverse
                // prompt might be tokenized with some following characters so
                // we'll compensate for that by widening the search window.
                for antiprompt in &args.antiprompt {
                    let extra_padding: usize = if interactive { 0 } else { 2 };
                    let search_start_pos = if last_output.len() > antiprompt.len() + extra_padding {
                        last_output.len() - antiprompt.len() - extra_padding
                    } else {
                        0
                    };
                    if last_output[search_start_pos..].contains(antiprompt.as_str()) {
                        if interactive {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens (:757-770)
                if !last_output.is_empty() {
                    if let Some(last_token) = sampler_last(sampler) {
                        for token in &antiprompt_token {
                            if *token == last_token {
                                if interactive {
                                    is_interacting = true;
                                }
                                is_antiprompt = true;
                                break;
                            }
                        }
                    }
                }
            }

            // deal with end of generation tokens in interactive mode (:777-795)
            let eog = sampler_last(sampler).is_some_and(|t| vocab.is_eog(t));
            if !waiting_for_first_input && eog {
                eprintln!("found an EOG token");

                if interactive {
                    if !args.antiprompt.is_empty() {
                        // tokenize and inject first reverse prompt (:782-787)
                        let first_antiprompt = vocab.tokenize(&args.antiprompt[0], false, true);
                        embd_inp.extend_from_slice(&first_antiprompt);
                        is_antiprompt = true;
                    }

                    if args.enable_chat_template {
                        if let Some(t) = templates.as_ref() {
                            let msg = std::mem::take(&mut assistant_ss);
                            chat_add_and_format(
                                t,
                                &mut chat_msgs,
                                "assistant",
                                msg,
                                vocab.add_bos,
                                vocab.add_eos,
                            );
                        }
                    }
                    is_interacting = true;
                    println!();
                }
            }

            if conversation_mode && !waiting_for_first_input && !prompt.is_empty() {
                // (:797-802)
                prompt.clear();
                is_interacting = false;
            }

            if (n_past > 0 || waiting_for_first_input) && is_interacting {
                // waiting for user input (:804-912)
                eprintln!("waiting for user input");

                if conversation_mode {
                    print!("\n> ");
                    let _ = std::io::stdout().flush();
                }

                if args.input_prefix_bos {
                    eprintln!("adding input prefix BOS token");
                    embd_inp.push(vocab.token_bos());
                }

                let mut buffer = String::new();
                if !args.input_prefix.is_empty() && !conversation_mode {
                    eprintln!("appending input prefix: '{}'", args.input_prefix);
                    print!("{}", args.input_prefix);
                    let _ = std::io::stdout().flush();
                }

                // color user input only (:822-824)
                console.set_display(DisplayType::UserInput);
                display = args.display_prompt;

                let mut line = String::new();
                loop {
                    // `console::readline` in a do/while (:826-831)
                    let another_line = console.readline(&mut line, args.multiline_input);
                    buffer.push_str(&line);
                    if !another_line {
                        break;
                    }
                }

                // done taking input, reset color (:833-835)
                console.set_display(DisplayType::Reset);
                display = true;

                if buffer.is_empty() {
                    // Ctrl+D on empty line exits (:837-840)
                    println!("EOF by user");
                    break;
                }

                if buffer.ends_with('\n') {
                    // Implement #587: (completion.cpp:842-848)
                    buffer.pop();
                }

                if buffer.is_empty() {
                    // Enter key on empty line lets the user pass control back
                    eprintln!("empty line, passing control back");
                } else {
                    // Add tokens to embd only if the input buffer is non-empty
                    // (:852-909)

                    // append input suffix if any (:853-857)
                    if !args.input_suffix.is_empty() && !conversation_mode {
                        eprintln!("appending input suffix: '{}'", args.input_suffix);
                        print!("{}", args.input_suffix);
                        let _ = std::io::stdout().flush();
                    }

                    if args.escape {
                        buffer = process_escapes(&buffer);
                    }

                    let format_chat = conversation_mode && args.enable_chat_template;
                    let user_inp = if format_chat {
                        let t = templates.as_ref().expect("conversation needs templates");
                        chat_add_and_format(
                            t,
                            &mut chat_msgs,
                            "user",
                            std::mem::take(&mut buffer),
                            vocab.add_bos,
                            vocab.add_eos,
                        )
                    } else {
                        buffer
                    };
                    // TODO(one inconvenient of current chat template
                    // implementation …, :871)
                    let line_pfx = vocab.tokenize(&args.input_prefix, false, true);
                    let line_inp = vocab.tokenize(&user_inp, false, format_chat);
                    let line_sfx = vocab.tokenize(&args.input_suffix, false, true);

                    // if user stop generation mid-way, we must add EOT to
                    // finish model's last response (:878-883)
                    if need_insert_eot && format_chat {
                        let eot = vocab.token_eot();
                        embd_inp.push(if eot == TOKEN_NULL { vocab.token_eos() } else { eot });
                        need_insert_eot = false;
                    }

                    embd_inp.extend_from_slice(&line_pfx);
                    embd_inp.extend_from_slice(&line_inp);
                    embd_inp.extend_from_slice(&line_sfx);

                    // reset assistant message (:904-905)
                    assistant_ss.clear();

                    n_remain -= line_inp.len() as i64;
                }

                input_echo = false; // do not echo this again
            }

            if n_past > 0 || waiting_for_first_input {
                // (:914-925)
                if is_interacting {
                    sampler.reset();
                }
                is_interacting = false;

                if waiting_for_first_input && args.single_turn {
                    interactive = false;
                    interactive_first = false;
                    INTERACTIVE_MODE.store(false, Ordering::SeqCst);
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation (:928-932)
        if !embd.is_empty() && vocab.is_eog(embd[embd.len() - 1]) && !interactive {
            println!(" [end of text]");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop
        // back to user input when reached (:934-939)
        if interactive && n_remain <= 0 && args.n_predict >= 0 {
            n_remain = args.n_predict as i64;
            is_interacting = true;
        }
    }

    // save session (:942-948)
    if !path_session.is_empty() && args.prompt_cache_all && !args.prompt_cache_ro {
        println!("llama_completion: saving final output to session file '{path_session}'");
        session_tokens.extend_from_slice(&embd);
        let mut bytes = Vec::with_capacity(8 + session_tokens.len() * 4);
        bytes.extend_from_slice(&(session_tokens.len() as u64).to_le_bytes());
        for t in &session_tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        let _ = std::fs::write(&path_session, &bytes);
        eprintln!(
            "saved final session to {path_session}, n_tokens = {}",
            session_tokens.len()
        );
    }

    // `LOG("\n\n")` + perf print (:950-951) — the perf summary is the port's
    // own diagnostics on stderr
    println!("\n");
    console.cleanup();
    0
}
