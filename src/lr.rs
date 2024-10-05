use burn::LearningRate;
use burn::lr_scheduler::LrScheduler;
use burn::prelude::Backend;

pub struct PolynomialDecay {
    start_lr: f32,
    end_lr: f32,
    power: f32,
    warmup_factor: f32,
    num_warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl PolynomialDecay {
    pub fn new(start_lr: f32, end_lr: f32, power: f32, total_steps: usize, num_warmup_steps: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            power,
            warmup_factor: 1.0 / num_warmup_steps as f32,
            total_steps,
            num_warmup_steps,
            current_step: 1,
        }
    }
}

impl<B: Backend> LrScheduler<B> for PolynomialDecay {
    type Record = (f32, f32, f32, f32, usize, usize, usize);

    fn step(&mut self) -> LearningRate {
        if self.current_step >= self.total_steps {
            return self.end_lr as f64;
        }

        if self.current_step <= self.num_warmup_steps {
            self.warmup_factor = self.current_step as f32 / self.num_warmup_steps as f32;
            self.current_step += 1;
            return (self.warmup_factor * self.start_lr) as f64;
        }

        let decay = (1.0 - (self.current_step - self.num_warmup_steps) as f32 / (self.total_steps - self.num_warmup_steps) as f32).powf(self.power);
        let lr = (self.start_lr - self.end_lr) * decay + self.end_lr;

        self.current_step += 1;

        lr as f64
    }

    fn to_record(&self) -> Self::Record {
        (
            self.start_lr,
            self.end_lr,
            self.power,
            self.warmup_factor,
            self.num_warmup_steps,
            self.total_steps,
            self.current_step
        )
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            start_lr: record.0,
            end_lr: record.1,
            power: record.2,
            warmup_factor: record.3,
            num_warmup_steps: record.4,
            total_steps: record.5,
            current_step: record.6,
        }
    }
}
