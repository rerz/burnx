use burn::prelude::{Backend, Bool, ElementConversion, Int, Tensor};
use burn::tensor::Distribution;
use rand::distributions::Uniform;
use rand::Rng;

pub trait MaskingStrategy<B: Backend> {
    type Config: Clone + Send + Sync;

    fn get_mask_indices(
        batch_and_seq_dims: [usize; 2],
        seq_lens: Vec<u32>,
        mask_config: Self::Config,
        device: &B::Device,
    ) -> Tensor<B, 2, Bool>;

    fn apply_mask(
        hidden: Tensor<B, 3>,
        mask_time_indices: Tensor<B, 2, Bool>,
        mask: Tensor<B, 1>,
        _attention_mask: Tensor<B, 2, Bool>,
        _device: &B::Device,
    ) -> Tensor<B, 3> {
        let mask_indices = mask_time_indices
            .unsqueeze_dim::<3>(2)
            .expand(hidden.shape());

        let mask = mask.expand(hidden.shape());

        let hidden = Tensor::mask_where(hidden, mask_indices, mask);

        hidden
    }
}


#[derive(Clone)]
pub struct NoneMask;

impl<B: Backend> MaskingStrategy<B> for NoneMask {
    type Config = ();

    fn get_mask_indices(batch_and_seq_dims: [usize; 2], seq_lens: Vec<u32>, mask_config: Self::Config, device: &B::Device) -> Tensor<B, 2, Bool> {
        Tensor::<B, 2, Int>::zeros(batch_and_seq_dims, device).bool()
    }
}

#[derive(Clone)]
pub struct AllMask;

impl<B: Backend> MaskingStrategy<B> for AllMask {
    type Config = ();

    fn get_mask_indices(batch_and_seq_dims: [usize; 2], seq_lens: Vec<u32>, mask_config: Self::Config, device: &B::Device) -> Tensor<B, 2, Bool> {
        Tensor::<B, 2, Int>::ones(batch_and_seq_dims, device).bool()
    }
}

#[derive(Clone)]
pub struct BlockMask;

#[derive(Clone)]
pub struct BlockMaskConfig {
    pub mask_prob: f32,
    pub mask_len: usize,
    pub min_masks: usize,
}

impl<B: Backend> MaskingStrategy<B> for BlockMask {
    type Config = BlockMaskConfig;

    fn get_mask_indices(
        [batch, seq]: [usize; 2],
        seq_lens: Vec<u32>,
        BlockMaskConfig {
            mask_len,
            mask_prob,
            min_masks,
        }: Self::Config,
        device: &B::Device,
    ) -> Tensor<B, 2, Bool> {

        assert!(mask_len < seq);

        let eps: f32 = rand::thread_rng().sample(Uniform::new(0.0, 1.0));
        let num_masked_spans = |input_len: usize| {
            let num_masked = (mask_prob * input_len as f32 / mask_len as f32 + eps) as usize;
            let mut num_masked = usize::max(num_masked, min_masks);

            if num_masked * mask_len > seq {
                num_masked = seq / mask_len;
            }

            if input_len - (mask_len - 1) < num_masked {
                num_masked = usize::max(input_len - (mask_len - 1), 0);
            }

            num_masked
        };

        let max_num_spans = num_masked_spans(seq);

        let mut mask = vec![];
        for seq_len in seq_lens {
            let num_masked_spans = num_masked_spans(seq_len as usize);

            let mut starting_indices = Tensor::<B, 1>::random(
                [num_masked_spans],
                Distribution::Uniform(0.0, (seq_len as usize - mask_len - 1) as f64),
                device,
            )
            .int();

            let dummy_idx = if starting_indices.dims()[0] == 0 {
                (seq - 1) as u32
            } else {
                starting_indices.to_data().to_vec::<B::IntElem>().unwrap()[0].elem::<u32>()
            };

            let padding = Tensor::ones(
                [max_num_spans.checked_sub(num_masked_spans).unwrap_or(0)],
                device,
            ) * dummy_idx;

            if padding.dims()[0] != 0 {
                starting_indices = Tensor::cat(vec![starting_indices, padding], 0);
            }

            let starting_indices = starting_indices.unsqueeze_dim::<2>(0);
            mask.push(starting_indices);
        }

        let mask = Tensor::cat(mask, 0);
        let mask = mask.unsqueeze_dim::<3>(2);
        let mask = mask.expand([batch, max_num_spans, mask_len]);
        let mask = mask.reshape([batch, max_num_spans * mask_len]);

        let offsets = Tensor::arange(0..mask_len as i64, device).unsqueeze_dims::<3>(&[0, 1]);
        let offsets = offsets.expand([batch, max_num_spans, mask_len]);
        let offsets = offsets.reshape([batch, max_num_spans * mask_len]);

        let mask = mask.clone() + offsets.clone();

        let mask_indices = Tensor::<B, 2, Int>::zeros([batch, seq], device);
        let mask_indices = Tensor::scatter(
            mask_indices,
            1,
            mask,
            Tensor::ones([batch, max_num_spans * mask_len], device),
        );

        mask_indices.bool()
    }
}


pub struct InverseBlockMask;


pub struct RandomMask {

}