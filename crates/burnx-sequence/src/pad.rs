pub enum SequencePad {
    SameLen,
    Explicit(usize)
}

pub fn sequences_to_tensor(sequences: Vec<Vec<f32>>, padding: SequencePad) {}