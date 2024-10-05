use burn::module::{ModuleVisitor, ParamId};
use burn::prelude::{Backend, ElementConversion, Tensor};
use burn::tensor::backend::AutodiffBackend;

pub fn l2<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
    let tensor = tensor.flatten::<1>(0, D - 1);
    let squared = tensor.powi_scalar(2);
    let summed = squared.sum();
    let norm = summed.sqrt();
    norm
}

pub fn cosine_similarity<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let dot = Tensor::sum_dim(a.clone() * b.clone(), dim);
    let norm_a = l2(a).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();
    let norm_b = l2(b).to_data().to_vec::<B::FloatElem>().unwrap()[0].elem::<f32>();

    let norm_a = f32::max(norm_a, 1e-8);
    let norm_b = f32::max(norm_b, 1e-8);

    let sim = dot / (norm_a * norm_b);

    sim
}

pub struct GradientMult<'a, B: AutodiffBackend> {
    pub multiplier: f32,
    pub grads: &'a mut B::Gradients,
}

impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientMult<'a, B> {
    fn visit_float<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grads) = tensor.grad(self.grads) {
            let multiplied = grads * self.multiplier;
            tensor.grad_replace(self.grads, multiplied);
        }
    }
}