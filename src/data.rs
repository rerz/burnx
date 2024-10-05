use burn::data::dataloader::Dataset;

pub trait MapDataset<I>: Dataset<I> {
    fn map<O, F: Fn(I) -> O>(self, mapper: F) -> MappedDataset<Self, I, O>;
}

pub struct MappedDataset<D, I, O> {
    inner: D,
    mapper: Box<dyn Fn(I) -> O>
}

impl<T, D, I> MapDataset<I> for T where T: Dataset<I> {
    fn map<O, F: Fn(I) -> O>(self, mapper: F) -> MappedDataset<Self, I, O> {
        MappedDataset {
            inner: self,
            mapper: Box::new(mapper),
        }
    }
}

impl<D, I, O> Dataset<I> for MappedDataset<D, I, O> where D: Dataset<I> {
    fn get(&self, index: usize) -> Option<I> {
        let mapped = self.mapper(self.inner.get(index).unwrap());
        Ok(mapped)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}