0.3.8

True

True



=== test_born_rule_sum ===

  [PASS] Born sum=1 [1x1->1] max_err=0.00e+00

  [PASS] Born P>=0  [1x1->1] min_P=2.50e-02

  [PASS] Born sum=1 [4x8->6] max_err=5.96e-08

  [PASS] Born P>=0  [4x8->6] min_P=1.18e-03

  [PASS] Born sum=1 [64x128->64] max_err=1.19e-07

  [PASS] Born P>=0  [64x128->64] min_P=0.00e+00

  [PASS] Born sum=1 [256x512->256] max_err=1.19e-07

  [PASS] Born P>=0  [256x512->256] min_P=-2.98e-08

  [PASS] Born sum=1 [1x1024->1024] max_err=1.19e-07

  [PASS] Born P>=0  [1x1024->1024] min_P=-1.49e-08



=== test_simplex_triangle_vertices ===

  [PASS] |+1> -> (1.0000, -0.5774), expected (1.0000, -0.5774)

  [PASS] |-1> -> (-1.0000, -0.5774), expected (-1.0000, -0.5774)

  [PASS] |0> -> (0.0000, 1.1547), expected (0.0000, 1.1547)

  [PASS] equilateral triangle: dist_diff=0.000000



=== test_centered_simplex_backward_fd ===

  [PASS] simplex backward v[0,0]: fd=-4.653931 analytic=-4.708171 rel_err=0.0115

  [PASS] simplex jacobian du identity: max_err=0.00e+00

  [PASS] simplex jacobian dv sqrt(3): max_err=0.00e+00



=== test_backnorm_full_fd ===

  [PASS] backnorm a_neg[0,0] fd=2.473831 an=2.471952

  [PASS] backnorm a_neg[3,4] fd=-0.438690 an=-0.440771

  [PASS] backnorm a_zero[1,2] fd=0.268936 an=0.267723

  [PASS] backnorm a_zero[7,5] fd=-1.441956 an=-1.444196

  [PASS] backnorm a_pos[2,1] fd=-0.699997 an=-0.698400

  [PASS] backnorm a_pos[6,3] fd=0.226974 an=0.226057

  [PASS] backnorm input[0,0] fd=-0.476837 an=-0.478536

  [PASS] backnorm input[2,5] fd=-2.052307 an=-2.054274



=== test_attention2_vs_sdpa ===

  [PASS] attn2 fwd [7x12] max_err=0.000000

  [PASS] attn2 dQ  [7x12] max_err=0.000002

  [PASS] attn2 dK  [7x12] max_err=0.000004

  [PASS] attn2 dV  [7x12] max_err=0.000001

  [PASS] attn2 fwd [32x64] max_err=0.000001

  [PASS] attn2 dQ  [32x64] max_err=0.000022

  [PASS] attn2 dK  [32x64] max_err=0.000038

  [PASS] attn2 dV  [32x64] max_err=0.000007

  [PASS] attn2 fwd [128x128] max_err=0.000001

  [PASS] attn2 dQ  [128x128] max_err=0.000036

  [PASS] attn2 dK  [128x128] max_err=0.000054

  [PASS] attn2 dV  [128x128] max_err=0.000012



=== test_attention2_batched_consistency ===

  [PASS] batched vs loop: max_err=0.00e+00



=== test_prior_raises_entropy ===

  [PASS] prior raised entropy: 0.0272 -> 1.0929

  [PASS] entropy above floor (~1.08): 1.0929



=== test_prior_no_overshoot ===

  [PASS] no overshoot: max_H=1.5850 <= 1.585

  [PASS] amplitudes finite after 500 prior steps



=== test_full_model_gradient_flow ===

  [PASS] all QTS layers receive gradients (0 dead)

  [PASS] token_emb receives gradient

  [PASS] pos_emb receives gradient

  [PASS] head.weight receives gradient



=== test_overfit_single_batch ===

  [PASS] overfit: loss dropped 5.563 -> 0.025 (ratio=0.0046)



=== test_checkpoint_roundtrip ===

  [PASS] checkpoint roundtrip: max_diff=0.00e+00



=== test_extreme_amplitude_scales ===

  [PASS] tiny amplitudes (1e-7): no NaN

  [PASS] tiny amplitudes (1e-7): no Inf

  [PASS] tiny amplitudes (1e-7): u in [-1,1]

  [PASS] small amplitudes (1e-4): no NaN

  [PASS] small amplitudes (1e-4): no Inf

  [PASS] small amplitudes (1e-4): u in [-1,1]

  [PASS] normal amplitudes (1.0): no NaN

  [PASS] normal amplitudes (1.0): no Inf

  [PASS] normal amplitudes (1.0): u in [-1,1]

  [PASS] large amplitudes (100): no NaN

  [PASS] large amplitudes (100): no Inf

  [PASS] large amplitudes (100): u in [-1,1]

  [PASS] huge amplitudes (1e4): no NaN

  [PASS] huge amplitudes (1e4): no Inf

  [PASS] huge amplitudes (1e4): u in [-1,1]



=== test_qitnn_linear_simplex_consistency ===

  [PASS] x matches

  [PASS] y matches



=== test_adamw_param_groups ===

  [PASS] QTS params found: 36

  [PASS] other params found: 14

  [PASS] QTS param wd=0 (got 0.0)

  [PASS] QTS param wd=0 (got 0.0)



=== test_generation_sanity ===

  [PASS] all generated tokens are valid ASCII (0 invalid)



=== test_mixed_precision_generation_sanity ===

  [PASS] mixed generation keeps bf16 visible activations

  [PASS] mixed generation returns token ids

  [PASS] mixed generation keeps ascii_guard valid (0 invalid)



=== test_mixed_precision_guardrails ===

  [PASS] mixed linear rejects half-cast QITNN master weights

  [PASS] mixed model rejects half-cast master weights

  [PASS] train() mixed path rejects missing bf16 support



=== test_mixed_precision_default_stays_fp32 ===

  [PASS] mixed_precision=False keeps q_proj output in fp32

  [PASS] mixed_precision=False keeps logits in fp32

  [PASS] mixed_precision=False keeps loss in fp32



=== test_qitnn_linear_mixed_precision_smoke ===

  [PASS] QITNNLinear mixed output is bf16

  [PASS] QITNNLinear master weights stay fp32

  [PASS] QITNNLinear mixed gradients are finite



=== test_qitnn_linear_mixed_precision_forces_bf16 ===

  [PASS] QITNNLinear mixed path ignores outer fp16 autocast for visible dtype

  [PASS] QITNNLinear mixed path keeps fp32 master-weight grads under outer fp16 autocast



=== test_model_mixed_precision_smoke ===

  [PASS] mixed_precision=True uses bf16 QTS activations

  [PASS] mixed_precision=True emits bf16 logits

  [PASS] mixed_precision=True keeps fp32 loss

  [PASS] mixed_precision=True keeps fp32 QTS weights

  [PASS] mixed_precision=True backward is finite



=== test_mixed_precision_cli_defaults ===

  [PASS] TrainConfig default resolves cleanly

  [PASS] CLI default preserves TrainConfig default

  [PASS] CLI legacy --mixed-precision maps to qts_fp32_rest_bf16

  [PASS] CLI legacy --no-mixed-precision resolves to fp32

  [PASS] CLI --precision-mode qts_fp32_rest_bf16 enables mixed path

  [PASS] CLI precision_mode alias normalizes to qts_fp32_rest_bf16

  [PASS] CLI conflicting legacy flag and precision_mode is rejected



=== test_precision_mode_high_level_api ===

  [PASS] precision_mode layer emits bf16 visible output

  [PASS] precision_mode model normalizes alias to qts_fp32_rest_bf16

  [PASS] precision_mode model drives bf16 q_proj activations

  [PASS] precision_mode model emits bf16 logits

  [PASS] precision_mode model keeps fp32 loss

  [PASS] precision_mode conflict with legacy bool is rejected



=== test_precision_mode_low_level_api ===

  [PASS] precision_mode forward3 bf16 outputs stay bf16

  [PASS] precision_mode forward3 raw channels stay fp32

  [PASS] precision_mode centered_simplex bf16 outputs stay bf16

  [PASS] precision_mode attention2 bf16 output stays bf16

  [PASS] precision_mode backward reaches bf16 activations

  [PASS] precision_mode backward keeps fp32 master-weight grads

  [PASS] precision_mode attention grads stay bf16

  [PASS] precision_mode prior_ accepts bf16 tensors

  [PASS] precision_mode conflict with legacy bool is rejected



=== test_native_mixed_bridge_smoke ===

  [PASS] native forward3 accepts bf16 input

  [PASS] native forward3 keeps cn/cz/cp in fp32

  [PASS] native centered_simplex preserves bf16

  [PASS] native attention2 preserves bf16

  [PASS] native attention backward preserves bf16

  [PASS] native prior accepts bf16 tensors



=== test_public_fp16_mixed_api_smoke ===

  [PASS] public forward3 fp16 outputs stay fp16

  [PASS] public forward3 raw channels stay fp32

  [PASS] public centered_simplex fp16 outputs stay fp16

  [PASS] public attention2 fp16 output stays fp16

  [PASS] public fp16 backward reaches activations

  [PASS] public fp16 backward keeps fp32 master-weight grads

  [PASS] public fp16 attention grads stay fp16

  [PASS] public prior_ accepts fp16 tensors



=== test_mixed_precision_training_parity_stress ===

  [PASS] mixed parity stress keeps mixed visible dtype pinned to bf16

  [PASS] mixed parity stress keeps both runs finite

  [PASS] mixed parity stress max train loss drift=0.0113

  [PASS] mixed parity stress eval loss gap=0.0526

  [PASS] mixed parity stress mean logit gap=0.1411

  [PASS] mixed parity stress max logit gap=0.7803

  [PASS] mixed parity stress q_proj diag gap=0.0021



=== test_memory_stability ===

  [PASS] memory growth after 100 steps: 0.00 MB



=== test_simplex_gelu_passthrough ===

  [PASS] y passes through unchanged

  [PASS] x = gelu(x_in)

  [PASS] grad flows to x half

  [PASS] grad flows to y half (=1.0)



=== test_init_near_uniform ===

  [PASS] init L0 wq: P=(0.332, 0.333, 0.334)

  [PASS] init L0 wk: P=(0.336, 0.331, 0.333)

  [PASS] init L0 wv: P=(0.335, 0.331, 0.334)

  [PASS] init L0 wo: P=(0.334, 0.335, 0.331)

  [PASS] init L0 ff1: P=(0.333, 0.338, 0.329)

  [PASS] init L0 ff2: P=(0.336, 0.333, 0.331)

  [PASS] init L1 wq: P=(0.334, 0.334, 0.332)

  [PASS] init L1 wk: P=(0.331, 0.336, 0.334)

  [PASS] init L1 wv: P=(0.335, 0.331, 0.334)

  [PASS] init L1 wo: P=(0.335, 0.329, 0.336)

  [PASS] init L1 ff1: P=(0.335, 0.332, 0.333)

  [PASS] init L1 ff2: P=(0.338, 0.332, 0.330)



=== test_determinism ===

  [PASS] deterministic: max loss diff between runs = 0.00e+00



=== test_residual_matters ===

  [PASS] pos_emb matters: normal=5.5498 vs zero=5.5446



=== test_byte_tokenizer_roundtrip ===

  [PASS] byte tokenizer vocab=256

  [PASS] byte tokenizer roundtrip exact



=== test_bpe_tokenizer_roundtrip ===

  [PASS] bpe tokenizer produced ids

  [PASS] bpe tokenizer vocab size

  [PASS] bpe tokenizer roundtrip exact



=== test_bpb_byte_math_helpers ===

  [PASS] byte target byte count equals token count

  [PASS] byte BPB from ln(2) loss equals 1.0

  [PASS] byte PPL from ln(2) loss equals 2.0

  [PASS] byte tokenizer keeps PPL = 2**BPB



=== test_bpb_bpe_byte_accounting ===

  [PASS] bpe byte accounting matches effective stream ['hello simple']

  [PASS] bpe byte accounting matches effective stream ['born rule at']

  [PASS] bpe byte accounting matches effective stream ['Привет simpl']



=== test_bpb_total_aggregation_math ===

  [PASS] window A BPB

  [PASS] window B BPB

  [PASS] exact total BPB uses total bits / total bytes

  [PASS] exact total BPB differs from naive window average



=== test_run_val_bpb_aggregation ===

  [PASS] run_val produced two windows

  [PASS] run_val keeps exact mean loss

  [PASS] run_val exact BPB uses total bits / total bytes

  [PASS] run_val BPB is not naive mean of window BPBs



=== test_eval_split_metrics_helper ===

  [PASS] eval_split_metrics windows

  [PASS] eval_split_metrics loss

  [PASS] eval_split_metrics bpb

  [PASS] eval_split_metrics ppl



=== test_bpe_trainer_smoke ===

tokenizer     bpe

vocab_size    320

data_format   auto

train_tokens  2930

val_tokens    398

params        33,792

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

saving        disabled (--no-save)

  [1] train_loss=5.82421875000  train_bpb=4.3368  train_ppl=338.3967  tok/s=333.6  lr=0.00030000

  [2] train_loss=5.79101562500  train_bpb=4.3828  train_ppl=327.3453  tok/s=515.6  lr=0.00003000

epoch 1/1  train_loss=5.79101562500  train_bpb=4.3828  train_ppl=327.3453  val_loss=5.7629394531  val_bpb=3.1894  val_ppl=318.2825

  steps train=2 val=24  tok/s train=515.0 val=3962.2  best_train=5.79101562500@1 (ppl=327.3453)  best_val=5.7629394531@1 (bpb=3.1894)  (ppl=318.2825)  time=0.1s

  [L0 ff1] (epoch 1)

    P-=0.345 P0=0.330 P+=0.325 | H=0.9801/1.5850 | eff=1.97/3 | col=12.5%

    ampl: sum_n=-15.9323 sum_z=4.3255 sum_p=-29.6967 | rms=0.456216/0.456530/0.446248

    dist: P0>0.4=34.6% P0<0.1=31.4% maxP>0.8=29.8% H>1.3=20.7% var(H)=0.1269

  [L0 ff2] (epoch 1)

    P-=0.342 P0=0.320 P+=0.337 | H=0.9835/1.5850 | eff=1.98/3 | col=13.4%

    ampl: sum_n=-13.6127 sum_z=0.2918 sum_p=-3.5099 | rms=0.452568/0.415394/0.438730

    dist: P0>0.4=35.3% P0<0.1=31.6% maxP>0.8=30.1% H>1.3=21.7% var(H)=0.1383

  [L0 wv] (epoch 1)

    P-=0.329 P0=0.328 P+=0.343 | H=0.9506/1.5850 | eff=1.93/3 | col=14.8%

    ampl: sum_n=-4.5235 sum_z=-5.8538 sum_p=6.1189 | rms=0.433259/0.438389/0.462949

    dist: P0>0.4=34.2% P0<0.1=33.2% maxP>0.8=33.4% H>1.3=18.8% var(H)=0.1389

  [L0 wo] (epoch 1)

    P-=0.350 P0=0.333 P+=0.317 | H=0.9811/1.5850 | eff=1.97/3 | col=14.8%

    ampl: sum_n=-2.7058 sum_z=6.3427 sum_p=1.9477 | rms=0.439711/0.430469/0.420184

    dist: P0>0.4=39.1% P0<0.1=32.2% maxP>0.8=28.9% H>1.3=23.2% var(H)=0.1454

first_loss 5.82421875

first_bpb 4.336811100736783

last_loss 5.7578125

last_bpb 4.4302760213965255

first_ppl 338.3966573919768

last_ppl 316.65488817567297

generated_text_begin

hello simplex rule�X�#��

generated_text_end

  [PASS] bpe trainer produced finite loss

  [PASS] bpe trainer returned finite train BPB

  [PASS] bpe trainer returned finite val BPB



=== test_mixed_precision_trainer_smoke ===

tokenizer     byte

vocab_size    256

data_format   auto

train_tokens  7143

val_tokens    793

params        29,632

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

saving        disabled (--no-save)

  [1] train_loss=5.49804687500  train_bpb=7.9320  train_ppl=244.2145  tok/s=840.5  lr=0.00030000

  [2] train_loss=5.55664062500  train_bpb=8.0165  train_ppl=258.9515  tok/s=893.1  lr=0.00003000

epoch 1/1  train_loss=5.55664062500  train_bpb=8.0165  train_ppl=258.9515  val_loss=5.5326052296  val_bpb=7.9819  val_ppl=252.8017

  steps train=2 val=49  tok/s train=891.1 val=4066.0  best_train=5.55664062500@1 (ppl=258.9515)  best_val=5.5326052296@1 (bpb=7.9819)  (ppl=252.8017)  time=0.0s

  [L0 ff1] (epoch 1)

    P-=0.345 P0=0.330 P+=0.325 | H=0.9816/1.5850 | eff=1.97/3 | col=12.2%

    ampl: sum_n=-9.7390 sum_z=3.5443 sum_p=-17.6768 | rms=0.449847/0.455198/0.440926

    dist: P0>0.4=34.4% P0<0.1=30.8% maxP>0.8=29.3% H>1.3=20.1% var(H)=0.1273

  [L0 ff2] (epoch 1)

    P-=0.343 P0=0.320 P+=0.337 | H=0.9842/1.5850 | eff=1.98/3 | col=13.9%

    ampl: sum_n=-13.9129 sum_z=1.2517 sum_p=2.6518 | rms=0.456485/0.416776/0.440023

    dist: P0>0.4=35.7% P0<0.1=32.0% maxP>0.8=28.9% H>1.3=22.1% var(H)=0.1383

  [L0 wv] (epoch 1)

    P-=0.329 P0=0.326 P+=0.345 | H=0.9471/1.5850 | eff=1.93/3 | col=16.2%

    ampl: sum_n=1.4265 sum_z=-4.8314 sum_p=3.0279 | rms=0.440213/0.439444/0.468014

    dist: P0>0.4=35.4% P0<0.1=33.8% maxP>0.8=33.4% H>1.3=18.2% var(H)=0.1427

  [L0 wo] (epoch 1)

    P-=0.348 P0=0.336 P+=0.316 | H=0.9826/1.5850 | eff=1.98/3 | col=14.8%

    ampl: sum_n=3.0354 sum_z=4.5951 sum_p=1.1266 | rms=0.444895/0.439066/0.424403

    dist: P0>0.4=38.3% P0<0.1=30.1% maxP>0.8=29.1% H>1.3=23.6% var(H)=0.1420

first_loss 5.498046875

first_bpb 7.9320049611375625

last_loss 5.615234375

last_bpb 8.101070786241738

first_ppl 244.21448474313812

last_ppl 274.57772580895397

generated_text_begin

PoAj?'>Fiplexgo 1%^

generated_text_end

  [PASS] mixed trainer produced finite loss

  [PASS] mixed trainer returns mixed model

  [PASS] mixed trainer preserves fp32 QTS master weights

  [PASS] mixed trainer preserves fp32 head weights

  [PASS] mixed trainer keeps bf16 visible logits after training

  [PASS] mixed trainer keeps fp32 loss after training

  [PASS] mixed trainer first BPB tracks byte loss

  [PASS] mixed trainer last BPB tracks byte loss

  [PASS] mixed trainer epoch train BPB tracks byte loss

  [PASS] mixed trainer epoch val BPB tracks byte loss



=== test_mixed_precision_checkpoint_resume_smoke ===

tokenizer     byte

vocab_size    256

data_format   auto

train_tokens  7085

val_tokens    787

params        29,632

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

run_dir       MyPath\_tmp_mixed_resume_runs\mixed_resume_a

csv_log       MyPath\_tmp_mixed_resume_runs\mixed_resume_a\metrics.csv

save_every    1  model_only=False

  [1] train_loss=5.56250000000  train_bpb=8.0250  train_ppl=260.4732  tok/s=1012.8  lr=0.00030000

  [2] train_loss=5.54199218750  train_bpb=7.9954  train_ppl=255.1859  tok/s=987.6  lr=0.00003000

epoch 1/1  train_loss=5.54199218750  train_bpb=7.9954  train_ppl=255.1859  val_loss=5.5549665179  val_bpb=8.0141  val_ppl=258.5183

  steps train=2 val=49  tok/s train=984.8 val=4111.8  best_train=5.54199218750@1 (ppl=255.1859)  best_val=5.5549665179@1 (bpb=8.0141)  (ppl=258.5183)  time=0.0s

  [L0 ff1] (epoch 1)

    P-=0.346 P0=0.331 P+=0.323 | H=0.9832/1.5850 | eff=1.98/3 | col=12.1%

    ampl: sum_n=-20.0939 sum_z=3.3921 sum_p=-17.2512 | rms=0.454209/0.454944/0.445556

    dist: P0>0.4=35.4% P0<0.1=30.1% maxP>0.8=29.3% H>1.3=21.6% var(H)=0.1252

  [L0 ff2] (epoch 1)

    P-=0.342 P0=0.320 P+=0.338 | H=0.9807/1.5850 | eff=1.97/3 | col=14.6%

    ampl: sum_n=-11.0124 sum_z=0.1544 sum_p=1.3775 | rms=0.452722/0.417900/0.438418

    dist: P0>0.4=35.2% P0<0.1=32.0% maxP>0.8=28.9% H>1.3=21.3% var(H)=0.1398

  [L0 wv] (epoch 1)

    P-=0.330 P0=0.326 P+=0.344 | H=0.9486/1.5850 | eff=1.93/3 | col=14.6%

    ampl: sum_n=-1.8392 sum_z=-2.2504 sum_p=5.3753 | rms=0.444155/0.445268/0.465032

    dist: P0>0.4=35.0% P0<0.1=32.8% maxP>0.8=33.0% H>1.3=18.2% var(H)=0.1427

  [L0 wo] (epoch 1)

    P-=0.347 P0=0.336 P+=0.317 | H=0.9829/1.5850 | eff=1.98/3 | col=14.3%

    ampl: sum_n=-0.2521 sum_z=5.0933 sum_p=4.1573 | rms=0.447110/0.443764/0.437668

    dist: P0>0.4=38.5% P0<0.1=30.3% maxP>0.8=28.5% H>1.3=23.8% var(H)=0.1436

  saved best -> MyPath\_tmp_mixed_resume_runs\mixed_resume_a\ckpt_best.pt

  saved MyPath\_tmp_mixed_resume_runs\mixed_resume_a\ckpt_ep1.pt

saved final -> MyPath\_tmp_mixed_resume_runs\mixed_resume_a\ckpt_final.pt

first_loss 5.5625

first_bpb 8.02499116494486

last_loss 5.521484375

last_bpb 7.965818126158398

first_ppl 260.4732060371668

last_ppl 250.00586435321858

generated_text_begin

hello simplexg4S<I%HP<m].>N$y

generated_text_end

resumed from MyPath\_tmp_mixed_resume_runs\mixed_resume_a\ckpt_final.pt  epoch=1  step=2

tokenizer     byte

vocab_size    256

data_format   auto

train_tokens  7085

val_tokens    787

params        29,632

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

run_dir       MyPath_tmp_mixed_resume_runs\mixed_resume_b

csv_log       MyPath\_tmp_mixed_resume_runs\mixed_resume_b\metrics.csv

save_every    1  model_only=False

  [1] train_loss=5.55664062500  train_bpb=8.0165  train_ppl=258.9515  tok/s=988.9  lr=0.00030000

epoch 2/2  train_loss=5.55664062500  train_bpb=8.0165  train_ppl=258.9515  val_loss=5.5438855230  val_bpb=7.9981  val_ppl=255.6695

  steps train=1 val=49  tok/s train=927.9 val=3418.6  best_train=5.55664062500@2 (ppl=258.9515)  best_val=5.5438855230@2 (bpb=7.9981)  (ppl=255.6695)  time=0.0s

  [L0 ff1] (epoch 2)

    P-=0.345 P0=0.332 P+=0.323 | H=0.9823/1.5850 | eff=1.98/3 | col=12.1%

    ampl: sum_n=-19.3920 sum_z=5.6233 sum_p=-14.0198 | rms=0.459701/0.458647/0.450335

    dist: P0>0.4=35.3% P0<0.1=30.7% maxP>0.8=29.3% H>1.3=20.5% var(H)=0.1249

  [L0 ff2] (epoch 2)

    P-=0.342 P0=0.320 P+=0.338 | H=0.9805/1.5850 | eff=1.97/3 | col=14.6%

    ampl: sum_n=-8.7040 sum_z=-1.5538 sum_p=0.5932 | rms=0.457841/0.422698/0.440741

    dist: P0>0.4=35.2% P0<0.1=32.5% maxP>0.8=28.9% H>1.3=21.2% var(H)=0.1397

  [L0 wv] (epoch 2)

    P-=0.329 P0=0.326 P+=0.344 | H=0.9497/1.5850 | eff=1.93/3 | col=14.6%

    ampl: sum_n=0.0838 sum_z=-0.4878 sum_p=4.5340 | rms=0.448383/0.448917/0.470034

    dist: P0>0.4=35.2% P0<0.1=32.0% maxP>0.8=33.0% H>1.3=18.4% var(H)=0.1435

  [L0 wo] (epoch 2)

    P-=0.347 P0=0.336 P+=0.317 | H=0.9827/1.5850 | eff=1.98/3 | col=14.3%

    ampl: sum_n=-1.8483 sum_z=6.9940 sum_p=6.3827 | rms=0.452663/0.451383/0.441982

    dist: P0>0.4=38.3% P0<0.1=30.1% maxP>0.8=28.5% H>1.3=24.2% var(H)=0.1438

  saved best -> MyPath\_tmp_mixed_resume_runs\mixed_resume_b\ckpt_best.pt

  saved MyPath\_tmp_mixed_resume_runs\mixed_resume_b\ckpt_ep2.pt

saved final -> MyPath\_tmp_mixed_resume_runs\mixed_resume_b\ckpt_final.pt

first_loss 5.556640625

first_bpb 8.01653787368965

last_loss 5.556640625

last_bpb 8.01653787368965

first_ppl 258.95145844419636

last_ppl 258.95145844419636

generated_text_begin

hello simplexg4+@IIL?I{]27w$F

generated_text_end

  [PASS] mixed checkpoint run saved final checkpoint

  [PASS] mixed checkpoint run saved tokenizer asset

  [PASS] mixed checkpoint metrics.csv carries BPB columns

  [PASS] mixed resume advanced global_step

  [PASS] mixed resume advanced epoch

  [PASS] mixed resumed model still matches final checkpoint after eval

  [PASS] mixed resumed model keeps fp32 QTS master weights

  [PASS] mixed resumed model keeps bf16 visible logits

  [PASS] mixed resumed model keeps fp32 loss



=== test_mixed_precision_cli_smoke ===

  [PASS] mixed CLI process exits cleanly

  [PASS] mixed CLI reports mixed mode enabled

  [PASS] mixed CLI reports precision_mode qts_fp32_rest_bf16

  [PASS] mixed CLI reports train BPB

  [PASS] mixed CLI reports val BPB

  [PASS] mixed CLI reaches generation output



=== test_json_loader_extracts_text ===

  [PASS] json loader captured text field

  [PASS] json loader captured nested content field

  [PASS] json loader emitted plain corpus text



=== test_json_trainer_smoke ===

tokenizer     byte

vocab_size    256

data_format   jsonl

train_tokens  3109

val_tokens    345

params        29,632

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

saving        disabled (--no-save)

  [1] train_loss=5.58789062500  train_bpb=8.0616  train_ppl=267.1715  tok/s=929.9  lr=0.00030000

  [2] train_loss=5.59082031250  train_bpb=8.0658  train_ppl=267.9553  tok/s=989.6  lr=0.00003000

epoch 1/1  train_loss=5.59082031250  train_bpb=8.0658  train_ppl=267.9553  val_loss=5.5593377976  val_bpb=8.0204  val_ppl=259.6508

  steps train=2 val=21  tok/s train=986.7 val=3164.1  best_train=5.59082031250@1 (ppl=267.9553)  best_val=5.5593377976@1 (bpb=8.0204)  (ppl=259.6508)  time=0.0s

  [L0 ff1] (epoch 1)

    P-=0.345 P0=0.330 P+=0.325 | H=0.9845/1.5850 | eff=1.98/3 | col=12.5%

    ampl: sum_n=-12.7605 sum_z=2.2419 sum_p=-27.9976 | rms=0.450614/0.454154/0.438999

    dist: P0>0.4=34.5% P0<0.1=30.2% maxP>0.8=29.6% H>1.3=21.3% var(H)=0.1271

  [L0 ff2] (epoch 1)

    P-=0.342 P0=0.321 P+=0.337 | H=0.9814/1.5850 | eff=1.97/3 | col=14.9%

    ampl: sum_n=-9.9607 sum_z=-0.6459 sum_p=4.2568 | rms=0.451009/0.420706/0.436379

    dist: P0>0.4=35.9% P0<0.1=31.7% maxP>0.8=28.9% H>1.3=21.4% var(H)=0.1396

  [L0 wv] (epoch 1)

    P-=0.331 P0=0.323 P+=0.346 | H=0.9456/1.5850 | eff=1.93/3 | col=15.8%

    ampl: sum_n=1.0275 sum_z=-2.6730 sum_p=3.6816 | rms=0.443743/0.445462/0.471475

    dist: P0>0.4=34.4% P0<0.1=34.0% maxP>0.8=32.4% H>1.3=17.2% var(H)=0.1422

  [L0 wo] (epoch 1)

    P-=0.349 P0=0.334 P+=0.317 | H=0.9802/1.5850 | eff=1.97/3 | col=13.3%

    ampl: sum_n=-0.6456 sum_z=6.4799 sum_p=4.5634 | rms=0.443704/0.439293/0.425407

    dist: P0>0.4=38.1% P0<0.1=30.7% maxP>0.8=29.3% H>1.3=23.4% var(H)=0.1399

first_loss 5.587890625

first_bpb 8.06162209371743

last_loss 5.59375

last_bpb 8.07007538497264

first_ppl 267.1714601120774

last_ppl 268.7415131590023

generated_text_begin

hello simplex^++:(IZ?'&=iPDeF

generated_text_end

  [PASS] json trainer produced finite loss



=== test_extended_dataset_best_final_test_metrics_smoke ===

tokenizer     byte

vocab_size    256

data_format   auto

train_tokens  2688

val_tokens    1664

test_tokens   1728

params        29,632

optimizer     adamw

mixed_prec    True

precision_mode qts_fp32_rest_bf16

lr            0.0003 -> 3e-05  (cosine)

seed          7

saving        disabled (--no-save)

  [1] train_loss=5.58593750000  train_bpb=8.0588  train_ppl=266.6502  tok/s=843.8  lr=0.00030000

epoch 1/2  train_loss=5.58593750000  train_bpb=8.0588  train_ppl=266.6502  val_loss=5.5529687500  val_bpb=8.0112  val_ppl=258.0024

  steps train=1 val=50  tok/s train=839.6 val=3732.4  best_train=5.58593750000@1 (ppl=266.6502)  best_val=5.5529687500@1 (bpb=8.0112)  (ppl=258.0024)  time=0.0s

  [L0 ff1] (epoch 1)

    P-=0.346 P0=0.329 P+=0.325 | H=0.9824/1.5850 | eff=1.98/3 | col=12.7%

    ampl: sum_n=-11.5754 sum_z=9.6777 sum_p=-23.8671 | rms=0.451708/0.451789/0.438099

    dist: P0>0.4=34.4% P0<0.1=30.0% maxP>0.8=28.6% H>1.3=20.5% var(H)=0.1284

  [L0 ff2] (epoch 1)

    P-=0.342 P0=0.319 P+=0.338 | H=0.9795/1.5850 | eff=1.97/3 | col=14.3%

    ampl: sum_n=-18.2570 sum_z=-2.7968 sum_p=1.4441 | rms=0.449232/0.411782/0.433296

    dist: P0>0.4=35.5% P0<0.1=32.5% maxP>0.8=29.1% H>1.3=21.4% var(H)=0.1402

  [L0 wv] (epoch 1)

    P-=0.331 P0=0.327 P+=0.343 | H=0.9444/1.5850 | eff=1.92/3 | col=15.4%

    ampl: sum_n=-0.1713 sum_z=-7.1646 sum_p=6.2214 | rms=0.442985/0.446923/0.470818

    dist: P0>0.4=34.8% P0<0.1=34.0% maxP>0.8=33.0% H>1.3=18.4% var(H)=0.1402

  [L0 wo] (epoch 1)

    P-=0.348 P0=0.336 P+=0.316 | H=0.9786/1.5850 | eff=1.97/3 | col=14.5%

    ampl: sum_n=-2.5099 sum_z=7.8742 sum_p=0.7892 | rms=0.443824/0.439611/0.427244

    dist: P0>0.4=39.3% P0<0.1=30.5% maxP>0.8=28.5% H>1.3=23.6% var(H)=0.1428

  [1] train_loss=5.53515625000  train_bpb=7.9855  train_ppl=253.4474  tok/s=973.7  lr=0.00003000

epoch 2/2  train_loss=5.53515625000  train_bpb=7.9855  train_ppl=253.4474  val_loss=5.5536718750  val_bpb=8.0123  val_ppl=258.1838

  steps train=1 val=50  tok/s train=968.4 val=4328.7  best_train=5.53515625000@2 (ppl=253.4474)  best_val=5.5529687500@1 (bpb=8.0112)  (ppl=258.0024)  time=0.0s

  [L0 ff1] (epoch 2)

    P-=0.346 P0=0.329 P+=0.325 | H=0.9824/1.5850 | eff=1.98/3 | col=12.7%

    ampl: sum_n=-12.8454 sum_z=9.0279 sum_p=-22.7393 | rms=0.452725/0.452041/0.439015

    dist: P0>0.4=34.4% P0<0.1=30.1% maxP>0.8=28.6% H>1.3=20.5% var(H)=0.1284

  [L0 ff2] (epoch 2)

    P-=0.342 P0=0.319 P+=0.338 | H=0.9796/1.5850 | eff=1.97/3 | col=14.3%

    ampl: sum_n=-18.3383 sum_z=-3.7205 sum_p=0.5106 | rms=0.449977/0.412514/0.434077

    dist: P0>0.4=35.5% P0<0.1=32.7% maxP>0.8=29.1% H>1.3=21.4% var(H)=0.1402

  [L0 wv] (epoch 2)

    P-=0.331 P0=0.327 P+=0.343 | H=0.9445/1.5850 | eff=1.92/3 | col=15.4%

    ampl: sum_n=-0.4334 sum_z=-7.9949 sum_p=6.6636 | rms=0.443144/0.448502/0.471240

    dist: P0>0.4=34.4% P0<0.1=34.2% maxP>0.8=33.0% H>1.3=18.2% var(H)=0.1402

  [L0 wo] (epoch 2)

    P-=0.348 P0=0.336 P+=0.316 | H=0.9785/1.5850 | eff=1.97/3 | col=14.5%

    ampl: sum_n=-1.8398 sum_z=7.1537 sum_p=0.6762 | rms=0.444827/0.440789/0.427273

    dist: P0>0.4=39.3% P0<0.1=30.5% maxP>0.8=28.5% H>1.3=23.6% var(H)=0.1426

final_test_loss     5.5536328125  final_test_bpb=8.0122  final_test_ppl=258.1738  (50 windows)

best_test_loss      5.5561328125  best_test_bpb=8.0158  best_test_ppl=258.8200  (50 windows)

first_loss 5.5859375

first_bpb 8.058804329965694

last_loss 5.53515625

last_bpb 7.985542472420551

first_ppl 266.6501501115634

last_ppl 253.44738569742526

generated_text_begin

hello simplexZ+++L%0?'.= {NeF

generated_text_end

  [PASS] extended trainer returned final_test_loss

  [PASS] extended trainer returned final_test_bpb

  [PASS] extended trainer returned final_test_ppl

  [PASS] extended trainer returned best_test_loss

  [PASS] extended trainer returned best_test_bpb

  [PASS] extended trainer returned best_test_ppl

  [PASS] legacy test_loss aliases final_test_loss

  [PASS] legacy test_bpb aliases final_test_bpb

  [PASS] legacy test_ppl aliases final_test_ppl



=== test_extended_dataset_cli_test_metrics_smoke ===

  [PASS] extended CLI test-metrics process exits cleanly

  [PASS] extended CLI prints final_test_loss

  [PASS] extended CLI prints best_test_loss



============================================================

STRESS TEST RESULTS: 219 passed, 0 failed, 0 warnings

Time: 21.8s

ALL STRESS TESTS PASSED