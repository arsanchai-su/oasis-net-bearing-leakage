| config                    |        n_params |   train_time_s |   infer_ms_per_window |   input_ms_per_window |
|:--------------------------|----------------:|---------------:|----------------------:|----------------------:|
| Features+GBM              |   nan           |         19.941 |                 0.026 |               nan     |
| OASIS v1 w/o adversary    |     2.83695e+07 |        142.595 |                 2.295 |                 2.452 |
| OASIS v1 w/o image branch | 53770           |          2.623 |                 0.031 |                 0.127 |
| OASIS v1 w/o order branch |     2.81192e+07 |        123.502 |                 2.296 |                 2.325 |
| OASIS w/o image branch    | 53770           |          2.008 |                 0.011 |                 0.127 |
| OASIS w/o order branch    |     2.81192e+07 |        145.235 |                 2.268 |                 2.325 |
| OASIS-Net (proposed)      |     2.83695e+07 |        142.082 |                 2.275 |                 2.452 |
| OASIS-Net v1 (a priori)   |     2.83695e+07 |        140.714 |                 2.302 |                 2.452 |
| Raw+WDCNN                 | 47910           |          2.454 |                 0.019 |               nan     |
| STFT+Hybrid               |     4.03509e+07 |        145.206 |                 2.74  |                 0.22  |
| STFT+ResNet18             |     1.11816e+07 |        497.748 |                 2.823 |                 0.22  |
| STFT+SwinT                |     2.7527e+07  |        121.017 |                 2.283 |                 0.22  |