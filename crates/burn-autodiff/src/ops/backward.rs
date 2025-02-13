use super::{Ops, OpsPrep};
use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::{ComputingProperty, NodeRef, Requirement},
    utils::duplicate,
};
use burn_tensor::backend::Backend;

/// Trait for all operations.
///
/// # Notes
///
/// Concrete types implementing this trait should not have any state.
/// If a state is necessary during the backward pass,
/// they should be declared with the associated type 'State'.
pub trait Backward<B, const N: usize>: Send + std::fmt::Debug
where
    Self: Sized + 'static,
    B: Backend,
{
    /// Associated type to compute the backward pass.
    type State: Clone + Send + std::fmt::Debug + 'static;

    /// The backward pass.
    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    );

    /// Prepare the backward ops.
    fn prepare<C: CheckpointStrategy>(
        self,
        nodes: [NodeRef; N],
    ) -> OpsPrep<Self, B, Self::State, C, N> {
        println!("Preparing {:?}", self);
        let requirement = Requirement::from_nodes(&nodes);
        OpsPrep::new(
            nodes,
            requirement,
            self,
            ComputingProperty::Ambiguous, // If not specified we start with ambiguous
            CheckpointerBuilder::default(),
        )
    }
}

/// Execute a binary operation during the backward step.
pub fn binary<B, FLhs, FRhs>(
    name: &str,
    parents: [Option<NodeRef>; 2],
    node: NodeRef,
    grads: &mut Gradients,
    func_lhs: FLhs,
    func_rhs: FRhs,
) where
    B: Backend,
    FLhs: FnOnce(B::FloatTensorPrimitive) -> B::FloatTensorPrimitive,
    FRhs: FnOnce(B::FloatTensorPrimitive) -> B::FloatTensorPrimitive,
{
    println!("{}", name);
    let [grad_4lhs, grad_4rhs] = duplicate(&parents, Some(grads.consume::<B>(&node)));
    let [node_lhs, node_rhs] = parents;

    if let Some(node) = node_lhs {
        println!("Before LHS: {:?}", grad_4lhs);
        let grad = func_lhs(grad_4lhs.unwrap());
        println!("After LHS: {:?}", grad);
        grads.register::<B>(node.id, grad)
    }

    if let Some(node) = node_rhs {
        println!("Before RHS: {:?}", grad_4rhs);
        let grad = func_rhs(grad_4rhs.unwrap());
        println!("After RHS: {:?}", grad);
        grads.register::<B>(node.id, grad)
    }
}

/// Execute a unary operation during the backward step.
pub fn unary<B, F>(name: &str, parents: [Option<NodeRef>; 1], node: NodeRef, grads: &mut Gradients, func: F)
where
    B: Backend,
    F: FnOnce(B::FloatTensorPrimitive) -> B::FloatTensorPrimitive,
{
    println!("{}", name);
    let [parent_node] = parents;
    let grad = grads.consume::<B>(&node);

    if let Some(node) = parent_node {
        println!("Before unary: {:?}", grad);
        let grad = func(grad);
        println!("After unary: {:?}", grad);
        grads.register::<B>(node.id, grad)
    }
}
