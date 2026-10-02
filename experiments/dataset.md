# Evaluation image set (frozen)

Frozen on 2026-10-02 at commit `66decc2` by `scripts/freeze_dataset.py`.
Resolution: 512 x 512, centre crop to square then `INTER_AREA` resize, lossless PNG in `data/eval_set/`.
The manifest with SHA-256 hashes is `data/eval_set/manifest.csv`. **Do not modify these files**; every table in the
report is computed on exactly this set (RESEARCH_PLAN.md section 3.5).

Rules: the painter receives these square PNGs directly, so the original code's stretch-to-square resize is the
identity and both implementations see identical pixels. Metrics (PSNR, SSIM, LPIPS) are computed at
512 x 512 against these files.

| # | Name | Origin | Source size (w x h) | Crop box (l, t, r, b) | Content | SHA-256 (first 12) |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `alien.png` | original repo test_images | 494 x 496 | 0, 1, 494, 495 | creature figure, dark glossy surfaces, white background | `b27fea176fe1` |
| 2 | `apple.png` | original repo test_images | 1021 x 854 | 83, 0, 937, 854 | still life, red apples in a wooden crate, wood texture | `b2eaf03b361c` |
| 3 | `cube2.png` | original repo test_images | 533 x 483 | 25, 0, 508, 483 | translucent coloured cubes, flat colour regions, sharp edges, white background | `b2a0606338cc` |
| 4 | `diamond.png` | original repo test_images | 396 x 377 | 9, 0, 386, 377 | faceted gem on dark background, specular highlights | `6566bdd114b7` |
| 5 | `diamond2.png` | original repo test_images | 477 x 463 | 7, 0, 470, 463 | glass block with iridescent edges, black background | `0a37166e4595` |
| 6 | `fire.png` | original repo test_images | 748 x 709 | 19, 0, 728, 709 | flames, high-frequency texture, dark background | `69d9b4ab937d` |
| 7 | `iceland.png` | original repo test_images | 568 x 452 | 58, 0, 510, 452 | landscape, waterfall and mountain at sunset, sky and water | `bc3f1b3a016c` |
| 8 | `jay.png` | original repo test_images | 729 x 640 | 44, 0, 684, 640 | portrait, face and hand, dark hair, shallow depth of field | `3ef962e9b6b8` |
| 9 | `joker.png` | original repo test_images | 559 x 563 | 0, 2, 559, 561 | face / portrait, painted make-up | `558810d1bcbe` |
| 10 | `sunflowers.png` | original repo test_images | 677 x 672 | 2, 0, 674, 672 | sunflowers in a basket, saturated yellow, blurred background | `affb33386246` |
| 11 | `yosemite.png` | original repo test_images | 640 x 480 | 80, 0, 560, 480 | landscape, rock face, waterfall and trees | `18169571240f` |
| 12 | `51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640.png` | self-collected | 426 x 640 | 0, 107, 426, 533 | minimal still life, pink balloon and white chair, flat pale background | `52ecdc676a77` |
| 13 | `53e2d3464b5bac14f1dc8460962e33791c3ad6e04e507440762e7ad39f4fc4_640.png` | self-collected | 640 x 427 | 106, 0, 533, 427 | urban scene, cafe chairs and tables, repeated geometric forms | `58764fbf0382` |
| 14 | `53e8d44b4253b10ff3d8992cc12c30771037dbf85254784e77267ed69649_640.png` | self-collected | 640 x 360 | 140, 0, 500, 360 | colour pencils in a radial pattern, saturated colours, sharp tips | `d0afb909a65c` |
| 15 | `54e2d3424c54a414f1dc8460962e33791c3ad6e04e507749712e79d29244c3_640.png` | self-collected | 640 x 426 | 107, 0, 533, 426 | macro still life, daisy and pencils, shallow depth of field | `027b9bf19d2f` |
| 16 | `54e4dd474257af14f1dc8460962e33791c3ad6e04e50744071297ad7954fc3_640.png` | self-collected | 640 x 426 | 107, 0, 533, 426 | stacked crates, dense saturated colour blocks, high-frequency texture | `9ccf9f9c9c7b` |
| 17 | `54e9d14b4e52a814f1dc8460962e33791c3ad6e04e50744172297ed39649c4_640.png` | self-collected | 640 x 433 | 103, 0, 536, 433 | animal, kitten in grass and daisies, shallow depth of field | `1ac5481e3928` |
| 18 | `55e4d4414a50ab14f1dc8460962e33791c3ad6e04e507441722a72dc9044cc_640.png` | self-collected | 640 x 512 | 64, 0, 576, 512 | dandelion seed on dark background, thin structures, reflection | `2a420363dcbe` |
| 19 | `57e3d6464e55aa14f1dc8460962e33791c3ad6e04e507440772d73d69545c6_640.png` | self-collected | 640 x 428 | 106, 0, 534, 428 | black-and-white checkerboard with a drain and water, high contrast | `d0b731b63f6b` |
| 20 | `57e4d0404d5baf14f1dc8460962e33791c3ad6e04e5074417d2e72d3964ec7_640.png` | self-collected | 640 x 360 | 140, 0, 500, 360 | cactus, repeated spines, green high-frequency texture | `05bb2d599c07` |
| 21 | `57e4d1474e5ba914f1dc8460962e33791c3ad6e04e5074417d2f7dd49f4ec6_640.png` | self-collected | 640 x 426 | 107, 0, 533, 426 | food still life, cereal bowl and spoon, warm colours | `6f1a4569ef39` |
| 22 | `57e5dd464d54ac14f1dc8460962e33791c3ad6e04e50744172297ed29f4bc4_640.png` | self-collected | 640 x 640 | 0, 0, 640, 640 | people on a vintage motorbike, faded retro colour grading | `2a41e340113f` |
| 23 | `57e8d7414852b10ff3d8992cc12c30771037dbf85254794e732f7bd29544_640.png` | self-collected | 640 x 426 | 107, 0, 533, 426 | abstract stacked paper layers, green and yellow diagonal bands | `c8ca1487bd25` |
| 24 | `ca-si-bui-truong-linh-1.png` | self-collected | 1000 x 1500 | 0, 250, 1000, 1250 | studio portrait, person in a green jacket, flat yellow background | `7541321d673e` |
| 25 | `emily-lau-NVi2yab124g-unsplash.png` | self-collected | 3456 x 5184 | 0, 864, 3456, 4320 | outdoor portrait, tinted glasses, blurred background | `83e50b660e62` |
| 26 | `ice-cream-cone-1274894_640.png` | self-collected | 640 x 426 | 107, 0, 533, 426 | still life, ice-cream cone, saturated blue background | `95b45f70e96a` |
| 27 | `maria-lysenko-3Bh0hy-yOcA-unsplash.png` | self-collected | 5574 x 3716 | 929, 0, 4645, 3716 | outdoor portrait, park background, shallow depth of field | `23ad4914447b` |
| 28 | `pexels-beratorer-30650522.png` | self-collected | 6336 x 9504 | 0, 1584, 6336, 7920 | macro flowers, yellow daisies, dark background | `fd404578eefb` |
| 29 | `pexels-christina99999-38524143.png` | self-collected | 4160 x 6240 | 0, 1040, 4160, 5200 | macro flowers, pink petals, fine detail | `3e1f7a222924` |
| 30 | `pexels-molnartamasphotography-29202983.png` | self-collected | 3072 x 4608 | 0, 768, 3072, 3840 | landscape, hazy hills and water, low contrast, low-frequency regions | `52df6e3f3e55` |

Total: 30 images (11 from the original repository, 19 self-collected).

## Self-collected photos

Photographs gathered by the student to cover portraits, landscapes, still life and high-texture scenes (target: 10 to
20). They are not part of the original repository; each keeps the file name it was collected under (`source_file` in
the manifest). The originals are in `data/raw_photos/`; the frozen 512 x 512 PNGs in `data/eval_set/` are the dataset,
so results do not need the originals. To add photos, drop them into `data/raw_photos/` and rerun
`scripts/freeze_dataset.py --force`; the repo images are re-generated byte-identically, so their hashes stay valid.
This must happen before the Week 5 ablations start; after that the set is closed.

## Preprocessing figure

`report/figures/preprocessing_pipeline.png` (from `scripts/make_preprocessing_figure.py`) shows input, crop,
normalized tensor and grid split for one image of this set.
