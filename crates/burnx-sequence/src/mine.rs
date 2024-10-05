use burn::prelude::Backend;

pub trait SamplingStrategy<B: Backend> {}

pub struct SampleWithinNegatives {}

pub struct SampleFromAnywhere {}