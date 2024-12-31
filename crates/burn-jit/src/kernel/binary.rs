use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_vectorization_factor,
};

use super::into_contiguous;

#[cube]
pub(crate) trait BinaryOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Line<C>, rhs: Line<C>) -> Line<C>;
}

pub(crate) struct AddOp;
pub(crate) struct SubOp;
pub(crate) struct MulOp;
pub(crate) struct DivOp;
pub(crate) struct RemainderOp;
pub(crate) struct PowOp;
pub(crate) struct AndOp; TODO() // Cubecl support for AND?

#[cube]
impl<N: Numeric> BinaryOp<N> for AddOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs + rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for SubOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs - rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for MulOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs * rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for DivOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs / rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for RemainderOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::rem(lhs, rhs)
    }
}

#[cube]
impl<N: Float> BinaryOp<N> for PowOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::powf(lhs, rhs)
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for AndOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs && rhs
    }
}

#[cube(launch)]
pub(crate) fn kernel_scalar_binop<C: Numeric, O: BinaryOp<C>>(
    input: &Tensor<Line<C>>,
    scalar: C,
    output: &mut Tensor<Line<C>>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    output[ABSOLUTE_POS] = O::execute(input[ABSOLUTE_POS], Line::new(scalar));
}

#[cube(launch)]
pub(crate) fn kernel_binop<C: Numeric, O: BinaryOp<C>>(
    lhs: &Tensor<Line<C>>,
    rhs: &Tensor<Line<C>>,
    out: &mut Tensor<Line<C>>,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous_lhs: bool,
    #[comptime] to_contiguous_rhs: bool,
) {
    let offset_out = ABSOLUTE_POS;
    let mut offset_lhs = ABSOLUTE_POS;
    let mut offset_rhs = ABSOLUTE_POS;

    if offset_out >= out.len() {
        return;
    }

    if to_contiguous_lhs {
        offset_lhs = index_offset_with_layout::<C, C>(
            lhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    if to_contiguous_rhs {
        offset_rhs = index_offset_with_layout::<C, C>(
            rhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    out[offset_out] = O::execute(lhs[offset_lhs], rhs[offset_rhs]);
}

pub(crate) fn launch_binop<R: JitRuntime, E: JitElement, O: BinaryOp<E>>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    let ndims = lhs.shape.num_dims();
    let vectorization_factor_lhs =
        tensor_vectorization_factor(&[4, 2], &lhs.shape.dims, &lhs.strides, ndims - 1);
    let vectorization_factor_rhs =
        tensor_vectorization_factor(&[4, 2], &rhs.shape.dims, &rhs.strides, ndims - 1);

    let vectorization_factor = Ord::min(vectorization_factor_lhs, vectorization_factor_rhs);

    let mut shape_out = vec![0; ndims];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let shape_out = Shape::from(shape_out);
    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    if lhs.can_mut_broadcast(&rhs) {
        kernel_binop::launch::<E, O, R>(
            &client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            TensorArg::alias(0),
            None,
            false,
            rhs.strides != lhs.strides || rhs.shape != lhs.shape,
        );

        lhs
    } else if rhs.can_mut_broadcast(&lhs) {
        kernel_binop::launch::<E, O, R>(
            &client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            TensorArg::alias(1),
            None,
            rhs.strides != lhs.strides || rhs.shape != lhs.shape,
            false,
        );

        rhs
    } else {
        let output = empty_device::<R, E>(lhs.client.clone(), lhs.device.clone(), shape_out);
        let to_contiguous_lhs = lhs.strides != output.strides || lhs.shape != output.shape;
        let to_contiguous_rhs = rhs.strides != output.strides || rhs.shape != output.shape;

        kernel_binop::launch::<E, O, R>(
            &client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            output.as_tensor_arg::<E>(vectorization_factor),
            None,
            to_contiguous_lhs,
            to_contiguous_rhs,
        );

        output
    }
}

pub(crate) fn launch_scalar_binop<R: JitRuntime, E: JitElement, O: BinaryOp<E>>(
    mut tensor: JitTensor<R>,
    scalar: E,
) -> JitTensor<R> {
    if !tensor.is_contiguous_buffer() {
        tensor = into_contiguous(tensor);
    }

    // Vectorization is only enabled when the last dimension is contiguous.
    let ndims = tensor.shape.num_dims();
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides, ndims - 1);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    if tensor.can_mut() {
        kernel_scalar_binop::launch::<E, O, R>(
            &client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(vectorization_factor),
            ScalarArg::new(scalar),
            TensorArg::alias(0),
        );

        tensor
    } else {
        let output = empty_device::<R, E>(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
        );

        kernel_scalar_binop::launch::<E, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            tensor.as_tensor_arg::<E>(vectorization_factor),
            ScalarArg::new(scalar),
            output.as_tensor_arg::<E>(vectorization_factor),
        );

        output
    }
}
