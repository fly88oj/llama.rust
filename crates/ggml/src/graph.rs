//! Compute graph — port of ggml.c `ggml_build_forward` / `ggml_cgraph`.

use crate::tensor::{Context, TensorId};

pub struct Graph {
    /// tensors in execution order (topological, parents first)
    pub nodes: Vec<TensorId>,
    /// input/leaf tensors (set via reset/add_leaf as in C)
    pub leafs: Vec<TensorId>,
    /// hash set of visited tensors to avoid duplicates (index into ctx)
    visited: Vec<bool>,
}

impl Graph {
    pub fn new(n_tensors_hint: usize) -> Self {
        Graph { nodes: Vec::with_capacity(n_tensors_hint), leafs: Vec::new(), visited: Vec::new() }
    }

    /// `ggml_build_forward_expand` — depth-first visit of srcs, then node.
    pub fn build_forward(&mut self, ctx: &Context, node: TensorId) {
        if self.visited.len() < ctx.tensors.len() {
            self.visited.resize(ctx.tensors.len(), false);
        }
        self.visit(ctx, node);
    }

    fn visit(&mut self, ctx: &Context, id: TensorId) {
        if self.visited[id.0 as usize] {
            return;
        }
        self.visited[id.0 as usize] = true;
        let t = &ctx.tensors[id.0 as usize];
        let srcs = t.src;
        for s in srcs.into_iter().flatten() {
            self.visit(ctx, s);
        }
        let op = ctx.tensors[id.0 as usize].op;
        if op != crate::tensor::GgmlOp::None {
            self.nodes.push(id);
        } else {
            // leaf
            self.leafs.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::GgmlOp;
    use crate::types::GgmlType;

    #[test]
    fn forward_order() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        let b = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        // fake ops: c = a+b ; d = c*a
        let c = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.tensors[c.0 as usize].op = GgmlOp::Add;
        let mut src = [None; crate::types::MAX_SRC];
        src[0] = Some(a);
        src[1] = Some(b);
        ctx.tensors[c.0 as usize].src = src;
        let d = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.tensors[d.0 as usize].op = GgmlOp::Mul;
        let mut src = [None; crate::types::MAX_SRC];
        src[0] = Some(c);
        src[1] = Some(a);
        ctx.tensors[d.0 as usize].src = src;

        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        assert_eq!(g.nodes, vec![c, d]);
        assert!(g.leafs.contains(&a) && g.leafs.contains(&b));
    }
}
