
/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/libggml-cpu.so:     file format elf64-x86-64


Disassembly of section .text:

000000000010fcc0 <ggml_gemv_mxfp4_8x8_q8_0+0xb0>:
  10fcc0:	mov    -0x28(%rsp),%rax
  10fcc5:	xor    %edi,%edi
  10fcc7:	mov    %r14,-0x20(%rsp)
  10fccc:	xor    %ecx,%ecx
  10fcce:	lea    (%rax,%r14,4),%rsi
  10fcd2:	mov    %r8,%r14
  10fcd5:	mov    %rdi,%rax
  10fcd8:	mov    %rdx,%r8
  10fcdb:	nopl   0x0(%rax,%rax,1)
  10fce0:	cmp    $0x1f,%r11d
  10fce4:	jle    110020 <ggml_gemv_mxfp4_8x8_q8_0+0x410>
  10fcea:	mov    0x3c22f(%rip),%rdi        # 14bf20 <ggml_table_f32_e8m0_half@@Base-0x1420>
  10fcf1:	imul   $0x88,%rax,%r13
  10fcf8:	mov    %r9,%rdx
  10fcfb:	vxorps %xmm1,%xmm1,%xmm1
  10fcff:	mov    %rcx,-0x8(%rsp)
  10fd04:	mov    %rax,-0x10(%rsp)
  10fd09:	add    %rbx,%r13
  10fd0c:	nopl   0x0(%rax)
  10fd10:	vpand  0x8(%r13),%ymm7,%ymm11
  10fd16:	vpandq 0x28(%r13),%ymm7,%ymm16
  10fd20:	vmovdqu 0x2(%rdx),%xmm3
  10fd25:	{evex} vpsrlw $0x4,0x8(%r13),%ymm5
  10fd30:	vpand  0x48(%r13),%ymm7,%ymm10
  10fd36:	vpand  0x68(%r13),%ymm7,%ymm14
  10fd3c:	vmovdqu 0x12(%rdx),%xmm2
  10fd41:	{evex} vpsrlw $0x4,0x28(%r13),%ymm13
  10fd4c:	vpand  %ymm7,%ymm5,%ymm5
  10fd50:	movzwl (%rdx),%eax
  10fd53:	{evex} vpsrlw $0x4,0x48(%r13),%ymm4
  10fd5e:	{evex} vpsrlw $0x4,0x68(%r13),%ymm12
  10fd69:	vpand  %ymm7,%ymm13,%ymm13
  10fd6d:	vpshufb %ymm5,%ymm6,%ymm5
  10fd72:	vpand  %ymm7,%ymm4,%ymm4
  10fd76:	vpand  %ymm7,%ymm12,%ymm12
  10fd7a:	vpshufb %ymm13,%ymm6,%ymm13
  10fd7f:	vpshufb %ymm4,%ymm6,%ymm4
  10fd84:	vpshufb %ymm12,%ymm6,%ymm12
  10fd89:	add    $0x88,%r13
  10fd90:	add    $0x22,%rdx
  10fd94:	vperm2i128 $0x0,%ymm3,%ymm3,%ymm3
  10fd9a:	vpshufb %ymm11,%ymm6,%ymm11
  10fd9f:	vpshufb %ymm16,%ymm6,%ymm16
  10fda5:	vpshufd $0xb1,%ymm16,%ymm15
  10fdac:	vpblendd $0xaa,%ymm15,%ymm11,%ymm15
  10fdb2:	vpshufd $0x0,%ymm3,%ymm0
  10fdb7:	vpshufd $0xb1,%ymm11,%ymm11
  10fdbd:	vpshufb %ymm10,%ymm6,%ymm10
  10fdc2:	vpsignb %ymm15,%ymm15,%ymm9
  10fdc7:	vpsignb %ymm15,%ymm0,%ymm15
  10fdcc:	vmovdqa32 %ymm19,%ymm0
  10fdd2:	vpshufb %ymm14,%ymm6,%ymm14
  10fdd7:	{vex} vpdpbusd %ymm15,%ymm9,%ymm0
  10fddc:	vpshufd $0x55,%ymm3,%ymm15
  10fde1:	vmovdqa32 %ymm16,%ymm9
  10fde7:	vperm2i128 $0x0,%ymm2,%ymm2,%ymm2
  10fded:	vmovq  %rax,%xmm17
  10fdf3:	vpblendd $0xaa,%ymm9,%ymm11,%ymm11
  10fdf9:	vmovq  %xmm17,%rcx
  10fdff:	vpsignb %ymm11,%ymm11,%ymm9
  10fe04:	vpsignb %ymm11,%ymm15,%ymm11
  10fe09:	vpshufd $0xaa,%ymm3,%ymm15
  10fe0e:	vpshufd $0xff,%ymm3,%ymm3
  10fe13:	{vex} vpdpbusd %ymm11,%ymm9,%ymm0
  10fe18:	vpshufd $0xb1,%ymm14,%ymm11
  10fe1e:	vpblendd $0xaa,%ymm11,%ymm10,%ymm11
  10fe24:	vpshufd $0xb1,%ymm10,%ymm10
  10fe2a:	vpblendd $0xaa,%ymm14,%ymm10,%ymm10
  10fe30:	vpsignb %ymm11,%ymm11,%ymm9
  10fe35:	vpsignb %ymm11,%ymm15,%ymm11
  10fe3a:	vpsignb %ymm10,%ymm3,%ymm3
  10fe3f:	{vex} vpdpbusd %ymm11,%ymm9,%ymm0
  10fe44:	vpsignb %ymm10,%ymm10,%ymm11
  10fe49:	vpshufd $0x0,%ymm2,%ymm10
  10fe4e:	{vex} vpdpbusd %ymm3,%ymm11,%ymm0
  10fe53:	vpshufd $0xb1,%ymm13,%ymm3
  10fe59:	vpblendd $0xaa,%ymm3,%ymm5,%ymm3
  10fe5f:	vpsignb %ymm3,%ymm3,%ymm11
  10fe64:	vpsignb %ymm3,%ymm10,%ymm3
  10fe69:	vpshufd $0x55,%ymm2,%ymm10
  10fe6e:	{vex} vpdpbusd %ymm3,%ymm11,%ymm0
  10fe73:	vpshufd $0xb1,%ymm5,%ymm3
  10fe78:	vpblendd $0xaa,%ymm13,%ymm3,%ymm3
  10fe7e:	vpsignb %ymm3,%ymm3,%ymm5
  10fe83:	vpsignb %ymm3,%ymm10,%ymm3
  10fe88:	{vex} vpdpbusd %ymm3,%ymm5,%ymm0
  10fe8d:	vpshufd $0xb1,%ymm12,%ymm3
  10fe93:	vpshufd $0xaa,%ymm2,%ymm5
  10fe98:	vpshufd $0xff,%ymm2,%ymm2
  10fe9d:	vpblendd $0xaa,%ymm3,%ymm4,%ymm3
  10fea3:	vpsignb %ymm3,%ymm3,%ymm10
  10fea8:	vpsignb %ymm3,%ymm5,%ymm3
  10fead:	movzbl -0x88(%r13),%eax
  10feb5:	{vex} vpdpbusd %ymm3,%ymm10,%ymm0
  10feba:	vpshufd $0xb1,%ymm4,%ymm3
  10febf:	vpblendd $0xaa,%ymm12,%ymm3,%ymm3
  10fec5:	vpsignb %ymm3,%ymm3,%ymm4
  10feca:	vpsignb %ymm3,%ymm2,%ymm2
  10fecf:	{vex} vpdpbusd %ymm2,%ymm4,%ymm0
  10fed4:	vmovq  %rax,%xmm2
  10fed9:	movzbl -0x84(%r13),%eax
  10fee1:	vcvtdq2ps %ymm0,%ymm0
  10fee5:	vmovq  %rax,%xmm5
  10feea:	movzbl -0x87(%r13),%eax
  10fef2:	vmovq  %rax,%xmm11
  10fef7:	movzbl -0x83(%r13),%eax
  10feff:	vmovq  %rax,%xmm10
  10ff04:	movzbl -0x86(%r13),%eax
  10ff0c:	vmovq  %rax,%xmm3
  10ff11:	movzbl -0x82(%r13),%eax
  10ff19:	vmovq  %rax,%xmm12
  10ff1e:	movzbl -0x85(%r13),%eax
  10ff26:	vmovq  %rax,%xmm4
  10ff2b:	movzbl -0x81(%r13),%eax
  10ff33:	vmovq  %rax,%xmm13
  10ff38:	vmovq  %xmm4,%rax
  10ff3d:	vmovss (%rdi,%rax,4),%xmm4
  10ff42:	vmovq  %xmm13,%rax
  10ff47:	vinsertps $0x10,(%rdi,%rax,4),%xmm4,%xmm4
  10ff4e:	vmovq  %xmm3,%rax
  10ff53:	vmovss (%rdi,%rax,4),%xmm3
  10ff58:	vmovq  %xmm12,%rax
  10ff5d:	vinsertps $0x10,(%rdi,%rax,4),%xmm3,%xmm3
  10ff64:	vmovq  %xmm11,%rax
  10ff69:	vmovlhps %xmm4,%xmm3,%xmm3
  10ff6d:	vmovss (%rdi,%rax,4),%xmm4
  10ff72:	vmovq  %xmm10,%rax
  10ff77:	vinsertps $0x10,(%rdi,%rax,4),%xmm4,%xmm4
  10ff7e:	vmovq  %xmm2,%rax
  10ff83:	vmovss (%rdi,%rax,4),%xmm2
  10ff88:	vmovq  %xmm5,%rax
  10ff8d:	vinsertps $0x10,(%rdi,%rax,4),%xmm2,%xmm2
  10ff94:	mov    0x3bf35(%rip),%rax        # 14bed0 <ggml_table_f32_f16@@Base-0x1870>
  10ff9b:	vmovlhps %xmm4,%xmm2,%xmm2
  10ff9f:	vinsertf128 $0x1,%xmm3,%ymm2,%ymm2
  10ffa5:	vmulps (%rax,%rcx,4){1to8},%ymm2,%ymm2
  10ffac:	vfmadd231ps %ymm2,%ymm0,%ymm1
  10ffb1:	cmp    %rdx,-0x18(%rsp)
  10ffb6:	jne    10fd10 <ggml_gemv_mxfp4_8x8_q8_0+0x100>
  10ffbc:	mov    -0x8(%rsp),%rcx
  10ffc1:	mov    -0x10(%rsp),%rax
  10ffc6:	vpermps %ymm1,%ymm8,%ymm1
  10ffcb:	inc    %rcx
  10ffce:	add    $0x20,%rsi
  10ffd2:	vmovups %ymm1,-0x20(%rsi)
  10ffd7:	add    %r10,%rax
  10ffda:	cmp    %r15,%rcx
  10ffdd:	.byte 0xf
  10ffde:	mov    %?,%ebp
