# Annotated Bibliography

Project: Photo-to-painting with a differentiable neural renderer (Nguyễn Lê Hoàng, VNUK, 2026-2027).

Conventions: entries are grouped by topic and ordered by year inside each group. "[verify]" marks a detail (page range, exact venue) to check against the publisher's page before the final report; everything else is stated with confidence. "Used in" names the report chapter(s) that cite the entry. Chapter names follow RESEARCH_PLAN.md: Related work, Theory, Method, Experiments, Limitations, Future work.

---

## 1. Stroke-based rendering (classical)

### [Haeberli 1990]
Haeberli, P. (1990). Paint by numbers: abstract image representations. *Proceedings of SIGGRAPH '90*, Computer Graphics 24(4), pp. 207–214.

Annotation: Introduces the idea of representing an image as an ordered list of brush strokes rather than a grid of pixels. Stroke position, size, orientation and color are sampled from the source photograph, the user steers placement interactively, and strokes are composited in order onto a canvas. This is the earliest formulation of stroke-based rendering (SBR) as "image as a stroke list", which is exactly the representation our optimizer outputs: an ordered array of stroke parameter vectors. It also establishes the color-sampling heuristic (take the stroke color from the photo at the stroke center) that the original Stylized Neural Painting sampler still uses to initialize strokes. We cite it to define SBR and to motivate why a vector representation is a different problem from pixel-wise image translation.

Used in: Related work, Theory.

### [Litwinowicz 1997]
Litwinowicz, P. (1997). Processing images and video for an impressionist effect. *Proceedings of SIGGRAPH '97*, pp. 407–414.

Annotation: Automates Haeberli's approach: strokes are placed on a regular grid, oriented along image gradients, clipped at strong edges and randomly perturbed, producing an impressionist look without user interaction. Extends the method to video by advecting strokes with optical flow so that they stay temporally coherent. The paper is the first fully automatic SBR pipeline and introduces gradient-based stroke orientation and edge clipping, hand-designed heuristics that our gradient-based optimization replaces with a learned objective. It is relevant to the time-lapse export of our system because it shows that stroke order and coherence matter for animation. We cite it as the automatic-heuristic baseline that learning-based methods improve upon.

Used in: Related work.

### [Hertzmann 1998]
Hertzmann, A. (1998). Painterly rendering with curved brush strokes of multiple sizes. *Proceedings of SIGGRAPH '98*, pp. 453–460.

Annotation: Introduces coarse-to-fine painting: the image is painted in layers with progressively smaller brushes, and each layer only paints where the current canvas differs from a blurred reference image by more than a threshold. Strokes are long curved B-splines that follow image gradients. This paper is the conceptual ancestor of two components of our pipeline: the error-map-driven stroke sampler (paint where the residual is large) and the progressive coarse-to-fine grid schedule (1x1 to MxM). Its "paint where the error is high" rule is the discrete version of the error-map probability distribution used by our sampler. We cite it in the theory chapter when deriving the progressive painter and in the related work as the classic multi-scale SBR algorithm.

Used in: Related work, Theory, Method.

### [Hertzmann 2003]
Hertzmann, A. (2003). A survey of stroke-based rendering. *IEEE Computer Graphics and Applications* 23(4), pp. 70–81.

Annotation: Surveys SBR up to 2003 and organizes the field into greedy algorithms, energy-minimization (optimization) approaches, and rendering by example. The energy-minimization view, where a painting is the stroke configuration that minimizes an objective over the canvas, is exactly the formulation of the neural painter: the differentiable renderer makes that energy minimizable by gradient descent instead of by heuristic or stochastic search. The survey also lists the open problems of the time (stroke ordering, style control, evaluation), several of which our metrics and ablations address directly. We use its taxonomy to structure the related work section and to position our work as the optimization branch with a learned renderer.

Used in: Related work.

---

## 2. Learning-based painting agents

### [Xie et al. 2013]
Xie, N., Hachiya, H., Sugiyama, M. (2013). Artist agent: a reinforcement learning approach to automatic stroke generation in oriental ink painting. *IEICE Transactions on Information and Systems* E96-D(5), pp. 1134–1144 [verify]. (An earlier version appeared at ICML 2012.)

Annotation: Trains a reinforcement learning (RL) agent to move a virtual brush and produce single ink strokes with realistic thickness and curvature. The policy is learned from reward signals rather than from a differentiable model, so gradients never flow through the renderer. This is an early example of the RL route to painting, which later work (Huang et al. 2019) scales to whole images. We cite it as evidence that stroke generation was treated as sequential decision making before differentiable renderers made direct optimization possible, and to contrast the exploration cost of RL with our gradient-based search.

Used in: Related work.

### [Ganin et al. 2018] (SPIRAL)
Ganin, Y., Kulkarni, T., Babuschkin, I., Eslami, S. M. A., Vinyals, O. (2018). Synthesizing programs for images using reinforced adversarial learning. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, pp. 1666–1675 [verify].

Annotation: SPIRAL trains an RL agent to emit drawing commands for a non-differentiable renderer (a real paint program), with an adversarial discriminator providing the reward. It shows that an agent can learn to reconstruct digits, characters and faces with a small number of strokes without any differentiable rendering. The cost is a very expensive training loop that must query the black-box renderer millions of times. The paper motivates the neural renderer idea: if the renderer can be approximated by a network, the expensive RL loop can be replaced by backpropagation. We cite it as the strongest non-differentiable baseline and as the motivation for surrogate renderers.

Used in: Related work.

### [Zheng et al. 2019] (StrokeNet)
Zheng, N., Jiang, Y., Huang, D. (2019). StrokeNet: a neural painting environment. *International Conference on Learning Representations (ICLR)*.

Annotation: Proposes learning a differentiable "painting environment" (a network that maps a stroke description to its rasterized image) and then training a drawing agent by backpropagating through that environment. The environment model is trained on synthetic strokes generated by a real painting program, the same on-the-fly synthetic training scheme our train_renderer.py uses. StrokeNet shows that the environment transfers across character and sketch datasets and that gradient-based training is far more sample-efficient than RL on the same task. It is a direct precursor of the neural renderer in Zou et al. and of our own renderer. Cited when justifying training the renderer from procedurally generated data rather than from real paintings.

Used in: Related work, Method.

### [Huang et al. 2019] (Learning to Paint)
Huang, Z., Heng, W., Zhou, S. (2019). Learning to paint with model-based deep reinforcement learning. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 8709–8718 [verify].

Annotation: Combines a neural stroke renderer (fully connected layers followed by pixel-shuffle upsampling) with a model-based deep RL actor that outputs quadratic Bézier strokes with variable width, color and opacity. Because the renderer is differentiable, the actor is trained with DDPG using gradients through the renderer, and a WGAN-GP discriminator supplies the reward. The system produces paintings with hundreds of strokes in a single forward pass at inference. The renderer architecture ("HuangNet" in the original Stylized Neural Painting code) is reused inside Zou et al.'s shape decoder and therefore inside ours. We cite it for the renderer design, for the Bézier parameterization used by our watercolor and marker brushes, and as the main RL-based competitor.

Used in: Related work, Method.

### [Liu et al. 2021] (Paint Transformer)
Liu, S., Lin, T., He, D., Li, F., Deng, R., Li, X., Ding, E., Wang, H. (2021). Paint Transformer: feed forward neural painting with stroke prediction. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6598–6607 [verify].

Annotation: Formulates painting as set prediction: a transformer predicts a set of strokes for a canvas-target pair in one forward pass and is trained on self-generated synthetic stroke data without any real painting dataset. It uses the same coarse-to-fine grid strategy as progressive rendering but replaces per-image optimization with a feed-forward network, making inference orders of magnitude faster. Its stroke rendering is a simple differentiable rasterizer for rectangular textured strokes, which limits the range of materials compared with a learned renderer. We cite it as the fastest competing approach and as the argument for why our per-image optimization trades speed for material flexibility and fidelity; the hardware benchmark is interpreted against the time budget it implies.

Used in: Related work, Experiments.

### [Schaldenbrand and Oh 2021]
Schaldenbrand, P., Oh, J. (2021). Content masked loss: human-like brush stroke planning in a reinforcement learning painting agent. *Proceedings of the AAAI Conference on Artificial Intelligence* 35(1), pp. 505–512 [verify].

Annotation: Modifies the reward of the Learning to Paint agent so that strokes are planned in a human-like order, salient content first and background later, using a content mask derived from a pretrained detector [verify: object detector vs. saliency]. It shows that changing the loss landscape changes stroke ordering without changing the renderer or the stroke model. Relevant to our work because stroke order determines the time-lapse animation, and because our error-map sampler is a cheap, detector-free way of prioritizing high-residual regions. We cite it when discussing stroke ordering and the limitations of our sampler.

Used in: Related work, Limitations.

---

## 3. Differentiable rendering and neural renderers

### [Nakano 2019] (Neural Painters)
Nakano, R. (2019). Neural Painters: a learned differentiable constraint for generating brushstroke paintings. arXiv:1904.08410.

Annotation: Trains generative networks (VAE and GAN variants) to imitate the MyPaint brush engine, obtaining a differentiable renderer for a real brush program, then optimizes stroke actions by gradient descent to reconstruct images and to maximize classifier activations ("painting what a network sees"). Introduces the framing of the neural painter as a differentiable constraint: any objective computable on pixels can be optimized over strokes once the renderer is differentiable. It also shows that stroke actions optimized through the approximate neural painter transfer back to the real brush engine. Our project adopts the same framing, and our renderer fidelity test (PSNR against the ground-truth rasterizer) follows this paper's evaluation. Cited in the theory chapter when explaining why a learned renderer is needed.

Used in: Related work, Theory.

### [Li et al. 2020] (DiffVG)
Li, T.-M., Lukáč, M., Gharbi, M., Ragan-Kelley, J. (2020). Differentiable vector graphics rasterization for editing and learning. *ACM Transactions on Graphics* 39(6), article 193 (Proceedings of SIGGRAPH Asia 2020).

Annotation: Provides an exact differentiable rasterizer for vector graphics (paths, Bézier curves, stroked outlines with anti-aliasing) by treating rasterization as an integral and deriving gradients with respect to shape parameters, including the discontinuities at edges. This is the analytic alternative to a learned neural renderer: gradients are exact, but the stroke model is limited to what the rasterizer supports, so a textured oil brush is out of reach. We cite it to explain the design space (analytic differentiable rasterizer versus neural surrogate) and to justify choosing a neural renderer for textured brushes. DiffVG is also the natural back end for the SVG export stretch goal, since our Bézier strokes map directly onto its path primitives.

Used in: Related work, Theory, Future work.

### [Kato et al. 2020]
Kato, H., Beker, D., Morariu, M., Ando, T., Matsuoka, T., Kehl, W., Gaidon, A. (2020). Differentiable rendering: a survey. arXiv:2006.12057.

Annotation: Surveys differentiable rendering for meshes, point clouds, implicit surfaces and voxels, and classifies methods by how they handle the non-differentiable rasterization step: approximate gradients, soft rasterization, or learned neural renderers. Although the survey focuses on 3D, its taxonomy applies directly to 2D stroke rendering, and it supplies the vocabulary (forward rendering, inverse rendering by gradient descent, surrogate gradients) used in our theory chapter. It also discusses the accuracy-versus-smoothness trade-off of approximate gradients, which we observe as the slight blur of the neural renderer. Cited to place neural stroke renderers within the broader differentiable rendering literature.

Used in: Related work, Theory.

### [Zou et al. 2021] (Stylized Neural Painting)
Zou, Z., Shi, T., Qiu, S., Yuan, Y., Shi, Z. (2021). Stylized Neural Painting. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15689–15698 [verify]. arXiv:2011.08114. Code: github.com/jiupinjia/stylized-neural-painting.

Annotation: The paper this project re-implements and extends. It frames image-to-painting as a stroke parameter search: a neural renderer (a dual-pathway network with a shape/alpha decoder and a color/texture decoder, "zou-fusion-net") is trained on procedurally rasterized strokes for four materials (oil paint, watercolor, marker pen, colored tape), and stroke parameters are then optimized against a target photo by gradient descent through the renderer. Its two technical contributions are the dual-pathway renderer, which renders textured strokes more faithfully than single-branch designs, and an optimal transport (Sinkhorn) loss that gives useful gradients when a stroke does not overlap its target region. It also introduces progressive coarse-to-fine grid rendering and joint optimization with neural style transfer. Our research questions RQ1 to RQ4 are ablations and extensions of this paper's design, and the original code with its pretrained checkpoints serves as the baseline in every results table.

Used in: Related work, Theory, Method, Experiments.

### [Kotovenko et al. 2021]
Kotovenko, D., Wright, M., Heimbrecht, A., Ommer, B. (2021). Rethinking style transfer: from pixels to parameterized brushstrokes. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12196–12205 [verify].

Annotation: Performs neural style transfer directly in the space of parameterized Bézier brushstrokes, using a lightweight differentiable renderer that blends strokes through distance-based soft assignment, and optimizes stroke parameters with the Gatys style loss. Shows that operating on strokes rather than pixels produces cleaner, more painterly stylizations and enables user control by editing individual strokes. This is closely related to the style-transfer stretch goal in our Week 9 plan, where the VGG Gram loss is applied to the rendered canvas and backpropagated into stroke parameters. Cited to compare their analytic renderer with our learned renderer and to support the claim that vector stroke spaces are a good domain for stylization.

Used in: Related work, Future work.

---

## 4. Optimal transport

### [Cuturi 2013]
Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. *Advances in Neural Information Processing Systems 26 (NIPS 2013)*, pp. 2292–2300.

Annotation: Adds an entropic regularization term to the optimal transport linear program, which turns it into a strictly convex problem solvable by Sinkhorn's matrix-scaling iterations, each a cheap matrix-vector product. The regularized distance is differentiable with respect to the input masses and converges orders of magnitude faster than exact solvers, which is what makes an OT term usable inside a gradient-descent loop. Our losses/sinkhorn.py implements exactly this algorithm (in the log domain for stability) on the pixel mass distributions of the canvas and the target. The theory chapter derives the Sinkhorn iterations from this paper and explains the roles of the regularization strength epsilon and the iteration count, both of which are swept in Ablation A.

Used in: Theory, Method, Experiments.

### [Peyré and Cuturi 2019]
Peyré, G., Cuturi, M. (2019). Computational Optimal Transport: with applications to data science. *Foundations and Trends in Machine Learning* 11(5–6), pp. 355–607.

Annotation: A book-length treatment of optimal transport for machine learning covering the Kantorovich formulation, Wasserstein distances, entropic regularization, the Sinkhorn algorithm with its log-domain and stabilized variants, and unbalanced transport. Chapter 4 gives the convergence analysis and the numerical pitfalls (underflow for small epsilon) that our implementation guards against. We use it as the reference for notation and for the statement that the entropic OT cost is a smooth approximation of the Wasserstein distance whose gradient with respect to the mass distributions is non-zero even when the two supports do not overlap, which is the property that fixes vanishing pixel-loss gradients. It is also the source for the debiased (normalized) Sinkhorn divergence option kept from the original code.

Used in: Theory.

---

## 5. Style transfer and perceptual metrics

### [Gatys et al. 2016]
Gatys, L. A., Ecker, A. S., Bethge, M. (2016). Image style transfer using convolutional neural networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2414–2423.

Annotation: Defines neural style transfer as optimizing an image so that its VGG feature activations match a content image while the Gram matrices of its activations match a style image. The Gram-matrix style loss is the loss used by the original Stylized Neural Painting for stroke-level style transfer and by our losses/style_loss.py stretch goal. The paper also establishes the optimization-based (per-image gradient descent) paradigm that the neural painter follows, with strokes instead of pixels as the variables. Cited in the theory chapter for the style loss equation and in the related work as the pixel-space stylization baseline.

Used in: Theory, Related work.

### [Wang et al. 2004] (SSIM)
Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing* 13(4), pp. 600–612.

Annotation: Introduces the structural similarity index (SSIM), which compares local luminance, contrast and structure statistics between two images instead of pixel-wise error, and correlates better with human judgments than PSNR or MSE. SSIM is one of our three fidelity metrics (with PSNR and LPIPS) for both renderer fidelity and painting fidelity, and its structural term is a candidate auxiliary loss for renderer training in Week 4. Cited in the experimental setup to define the metric and to justify reporting it alongside PSNR.

Used in: Experiments.

### [Zhang et al. 2018] (LPIPS)
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 586–595.

Annotation: Shows that distances between deep network activations (AlexNet, VGG, SqueezeNet) calibrated on human perceptual judgments predict perceived similarity far better than PSNR or SSIM, and releases the LPIPS metric. Paintings are deliberately not pixel-accurate, so a perceptual metric is needed to compare methods fairly; LPIPS is our primary perceptual metric in every results table and the fallback when the aesthetic survey has too few raters. Cited in the experimental setup and in the interpretation of ablation results where PSNR and LPIPS disagree.

Used in: Experiments.

---

## Summary

| Group | Entries |
| :--- | ---: |
| Stroke-based rendering (classical) | 4 |
| Learning-based painting agents | 6 |
| Differentiable rendering and neural renderers | 5 |
| Optimal transport | 2 |
| Style transfer and perceptual metrics | 3 |
| **Total** | **20** |

Required entries: 15. Added: Hertzmann 2003, Xie et al. 2013, Zheng et al. 2019, Schaldenbrand and Oh 2021, Kotovenko et al. 2021.
