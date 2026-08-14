/* NeuralAtlas glossary — the reference layer behind every selector.
 *
 * Every entry fills the SAME slots, and the panel draws those slots in the same
 * place every time: what it is, what makes it different from the rest, and a
 * fixed row of facts. Switching entries swaps the values and moves nothing —
 * the way a telemetry readout changes numbers without changing its dials.
 *
 * That constraint is the schema. `facts` is an ordered list of [label, value],
 * and within a kind every entry MUST carry the same labels in the same order,
 * or the panel will jump when you switch. FACT_KEYS below is the contract.
 *
 * Method ids carry the segmentation in parentheses ("Lime (SLIC)"), so lookup
 * strips it and falls back to the base algorithm.
 */

const CAPTUM = 'https://captum.ai/api/';

// The fixed dial faces, per kind. Same labels, same order, every entry.
export const FACT_KEYS = {
  model: ['Family', 'Params', 'Year', 'Depth', 'Read-out'],
  dataset: ['Images', 'Classes', 'Source', 'Preprocessing'],
  method: ['Family', 'Needs', 'Cost', 'Resolution', 'Caveat'],
  metric: ['Measures', 'Better', 'Removes', 'Caveat'],
};

export const WIKI_SECTIONS = [
  {
    kind: 'model',
    label: 'Models',
    entries: [
      {
        id: 'alexnet',
        title: 'AlexNet',
        tags: ['CNN', '2012'],
        summary: 'The eight-layer CNN that won ILSVRC 2012 and started the deep-learning era in vision.',
        differs: 'Throws spatial detail away early — an 11x11 stride-4 first kernel — so its maps come out blockier than the others.',
        facts: [
          ['Family', 'CNN'],
          ['Params', '61.1M'],
          ['Year', '2012'],
          ['Depth', '5 conv + 3 fully connected'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'torchvision: AlexNet', href: 'https://docs.pytorch.org/vision/main/models/alexnet.html' },
        ],
      },
      {
        id: 'resnet18',
        title: 'ResNet-18',
        tags: ['CNN', 'Residual'],
        summary: 'Eighteen layers with skip connections, so the gradient reaches the input without vanishing.',
        differs: 'Residual blocks: each one adds its input back to its output. That short path from logit to pixels is why gradient methods stay comparatively clean here.',
        facts: [
          ['Family', 'CNN, residual'],
          ['Params', '11.7M'],
          ['Year', '2015'],
          ['Depth', '18 layers'],
          ['Read-out', 'layer4 (7x7)'],
        ],
        links: [
          { label: 'Paper (arXiv 1512.03385)', href: 'https://arxiv.org/abs/1512.03385' },
          { label: 'torchvision: ResNet', href: 'https://docs.pytorch.org/vision/main/models/resnet.html' },
        ],
      },
      {
        id: 'efficientnet_b0',
        title: 'EfficientNet-B0',
        tags: ['CNN', 'NAS'],
        summary: 'The baseline of the EfficientNet family: the best accuracy per parameter of the three models here.',
        differs: 'Squeeze-and-excitation reweights whole channels from global context, so part of what it uses is not tied to any location — the maps look emptier than the accuracy suggests.',
        facts: [
          ['Family', 'CNN, found by NAS'],
          ['Params', '5.3M'],
          ['Year', '2019'],
          ['Depth', 'Inverted residual blocks'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'Paper (arXiv 1905.11946)', href: 'https://arxiv.org/abs/1905.11946' },
          { label: 'torchvision: EfficientNet', href: 'https://docs.pytorch.org/vision/main/models/efficientnet.html' },
        ],
      },
    ],
  },
  {
    kind: 'dataset',
    label: 'Datasets',
    entries: [
      {
        id: 'imagenet-pico',
        title: 'imagenet-pico',
        tags: ['Real photos'],
        summary: 'A small slice of the ImageNet-1k validation set — real photographs, original class ids and labels.',
        differs: 'The reference set. Whatever a method does here is its behaviour on the data the models were actually trained for.',
        facts: [
          ['Images', 'A handful per class'],
          ['Classes', '1000 (ImageNet-1k label space)'],
          ['Source', 'ImageNet-1k validation split'],
          ['Preprocessing', 'Resize 256, centre-crop 224'],
        ],
        links: [
          { label: 'ImageNet', href: 'https://www.image-net.org/' },
        ],
      },
      {
        id: 'imagenet-pico-ai',
        title: 'imagenet-pico-ai',
        tags: ['Generated'],
        summary: 'The same classes as imagenet-pico, but every image is generated rather than photographed.',
        differs: 'Built to be read against imagenet-pico: same labels, same viewer, synthetic inputs. A probe, not a benchmark — accuracy here is not comparable to a published ImageNet number.',
        facts: [
          ['Images', 'Paired with imagenet-pico'],
          ['Classes', 'The same 1000'],
          ['Source', 'Generated'],
          ['Preprocessing', 'Resize 256, centre-crop 224'],
        ],
        links: [],
      },
    ],
  },
  {
    kind: 'method',
    label: 'Methods',
    entries: [
      {
        id: 'Saliency',
        title: 'Saliency',
        tags: ['Gradient'],
        summary: 'The absolute gradient of the class score with respect to each input pixel.',
        differs: 'The cheapest there is, and the most literal: it answers "which pixel, nudged, would move the score most" — a slope, not a contribution.',
        facts: [
          ['Family', 'Gradient'],
          ['Needs', 'One backward pass'],
          ['Cost', 'Cheapest'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'Speckled — neighbours get very different gradients'],
        ],
        links: [
          { label: 'Paper (arXiv 1312.6034)', href: 'https://arxiv.org/abs/1312.6034' },
          { label: 'Captum: Saliency', href: `${CAPTUM}saliency.html` },
        ],
      },
      {
        id: 'InputXGradient',
        title: 'Input x Gradient',
        tags: ['Gradient'],
        summary: 'The gradient multiplied by the input value — a first-order estimate of each pixel\'s contribution.',
        differs: 'Scaling by the input turns a sensitivity into something closer to a contribution: a pixel matters if the model is sensitive to it and it is actually there.',
        facts: [
          ['Family', 'Gradient'],
          ['Needs', 'One backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'Still a point estimate — inherits the gradient noise'],
        ],
        links: [
          { label: 'Captum: InputXGradient', href: `${CAPTUM}input_x_gradient.html` },
        ],
      },
      {
        id: 'IntegratedGradients',
        title: 'Integrated Gradients',
        tags: ['Gradient', 'Axiomatic'],
        summary: 'Averages gradients along a straight path from a baseline to the input.',
        differs: 'The only one here with completeness: the attributions add up to the score difference between input and baseline.',
        facts: [
          ['Family', 'Gradient, axiomatic'],
          ['Needs', 'Gradients along a path from a baseline'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'A black baseline cannot credit dark evidence'],
        ],
        links: [
          { label: 'Paper (arXiv 1703.01365)', href: 'https://arxiv.org/abs/1703.01365' },
          { label: 'Captum: IntegratedGradients', href: `${CAPTUM}integrated_gradients.html` },
        ],
      },
      {
        id: 'LayerIntegratedGradients',
        title: 'Layer Integrated Gradients',
        tags: ['Gradient', 'Layer'],
        summary: 'Integrated Gradients computed on an internal layer\'s activations instead of on the pixels.',
        differs: 'Same path integral, but it lands on a feature map and is upsampled back — coarser than the input-space version, and smoother to look at.',
        facts: [
          ['Family', 'Gradient, layer'],
          ['Needs', 'Gradients along a path, at one layer'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Feature map, upsampled'],
          ['Caveat', 'The chosen layer sets what can be seen'],
        ],
        links: [
          { label: 'Captum: Layer attribution', href: `${CAPTUM}layer.html` },
        ],
      },
      {
        id: 'DeepLift',
        title: 'DeepLIFT',
        tags: ['Gradient', 'Baseline'],
        summary: 'Propagates the difference from a baseline activation backwards with per-layer rules.',
        differs: 'Sidesteps saturation: a confident neuron with zero gradient still passes its evidence back, which a raw gradient cannot do.',
        facts: [
          ['Family', 'Gradient, baseline'],
          ['Needs', 'One modified backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'The reference activation is a real choice'],
        ],
        links: [
          { label: 'Paper (arXiv 1704.02685)', href: 'https://arxiv.org/abs/1704.02685' },
          { label: 'Captum: DeepLift', href: `${CAPTUM}deep_lift.html` },
        ],
      },
      {
        id: 'DeepLiftShap',
        title: 'DeepLIFT SHAP',
        tags: ['Gradient', 'Shapley'],
        summary: 'DeepLIFT averaged over a distribution of baselines, approximating SHAP values.',
        differs: 'Averaging over many baselines turns the single-reference result into a Shapley estimate — one pass per baseline.',
        facts: [
          ['Family', 'Gradient, Shapley'],
          ['Needs', 'One pass per baseline'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'Cost grows with the baseline set'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: DeepLiftShap', href: `${CAPTUM}deep_lift_shap.html` },
        ],
      },
      {
        id: 'GradientShap',
        title: 'Gradient SHAP',
        tags: ['Gradient', 'Shapley'],
        summary: 'Expected gradients: samples noisy points between baselines and the input, then averages input x gradient.',
        differs: 'The added sampling noise is what smooths out the speckle plain Saliency shows.',
        facts: [
          ['Family', 'Gradient, Shapley'],
          ['Needs', 'Gradients at sampled noisy points'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'An estimate — two runs will not match exactly'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: GradientShap', href: `${CAPTUM}gradient_shap.html` },
        ],
      },
      {
        id: 'GuidedBackprop',
        title: 'Guided Backpropagation',
        tags: ['Gradient', 'Sharp'],
        summary: 'Backprop with negative gradients clipped at every ReLU — very sharp, barely class-sensitive.',
        differs: 'Only positive evidence flows backwards, so the maps look like clean edges. Convincing, and largely the same whichever class you ask about.',
        facts: [
          ['Family', 'Gradient, sharp'],
          ['Needs', 'One clipped backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'Barely changes with the class — says more about the image than the decision'],
        ],
        links: [
          { label: 'Paper (arXiv 1412.6806)', href: 'https://arxiv.org/abs/1412.6806' },
          { label: 'Captum: GuidedBackprop', href: `${CAPTUM}guided_backprop.html` },
        ],
      },
      {
        id: 'Deconvolution',
        title: 'Deconvolution',
        tags: ['Gradient', 'Sharp'],
        summary: 'Zeiler & Fergus reconstruction: at each ReLU the backward pass uses the sign of the gradient, not the forward activation.',
        differs: 'The oldest one here and the one that behaves worst under sanity checks — it survives randomising the weights far more than an explanation should.',
        facts: [
          ['Family', 'Gradient, sharp'],
          ['Needs', 'One modified backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'Kept as the historical reference, not because it should be trusted'],
        ],
        links: [
          { label: 'Paper (arXiv 1311.2901)', href: 'https://arxiv.org/abs/1311.2901' },
          { label: 'Sanity checks (arXiv 1810.03292)', href: 'https://arxiv.org/abs/1810.03292' },
        ],
      },
      {
        id: 'LayerGradCam',
        title: 'Grad-CAM',
        tags: ['Gradient', 'Layer'],
        summary: 'Weights a convolutional layer\'s feature maps by their average gradient — coarse but class-discriminative.',
        differs: 'The one to reach for when the question is whether the model looked at the object or at the background. It answers "where" well and "which pixel" not at all.',
        facts: [
          ['Family', 'Gradient, layer'],
          ['Needs', 'One backward pass at the last conv layer'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Feature map (7x7 on ResNet-18)'],
          ['Caveat', 'Upsampled to image size — the detail is invented'],
        ],
        links: [
          { label: 'Paper (arXiv 1610.02391)', href: 'https://arxiv.org/abs/1610.02391' },
          { label: 'Captum: LayerGradCam', href: `${CAPTUM}layer.html` },
        ],
      },
      {
        id: 'GuidedGradCam',
        title: 'Guided Grad-CAM',
        tags: ['Gradient', 'Hybrid'],
        summary: 'Element-wise product of Guided Backprop and Grad-CAM: the localisation of one, the detail of the other.',
        differs: 'The most persuasive-looking map in the set — sharp and well placed. Only the placement depends on the class.',
        facts: [
          ['Family', 'Gradient, hybrid'],
          ['Needs', 'Grad-CAM times Guided Backprop'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Caveat', 'The fine detail inherits Guided Backprop\'s weak class sensitivity'],
        ],
        links: [
          { label: 'Paper (arXiv 1610.02391)', href: 'https://arxiv.org/abs/1610.02391' },
          { label: 'Captum: GuidedGradCam', href: `${CAPTUM}guided_grad_cam.html` },
        ],
      },
      {
        id: 'Occlusion',
        title: 'Occlusion',
        tags: ['Perturbation'],
        summary: 'Slides a grey patch over the image and measures how far the class score drops.',
        differs: 'No gradients at all. Because it only ever measures real forward passes, it is the useful sanity check on everything above.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One forward pass per patch position'],
          ['Cost', 'Slow'],
          ['Resolution', 'The patch size'],
          ['Caveat', 'The patch size sets what can be resolved'],
        ],
        links: [
          { label: 'Paper (arXiv 1311.2901)', href: 'https://arxiv.org/abs/1311.2901' },
          { label: 'Captum: Occlusion', href: `${CAPTUM}occlusion.html` },
        ],
      },
      {
        id: 'Lime',
        title: 'LIME',
        tags: ['Perturbation', 'Superpixels'],
        summary: 'Fits a sparse linear surrogate on perturbed copies of the image, over superpixel segments.',
        differs: 'Explains a stand-in model, not the network: the attributions are the weights of a linear fit over segments switched on and off.',
        facts: [
          ['Family', 'Perturbation, superpixels'],
          ['Needs', 'A few hundred forward passes'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per segment'],
          ['Caveat', 'The segmentation does much of the work'],
        ],
        links: [
          { label: 'Paper (arXiv 1602.04938)', href: 'https://arxiv.org/abs/1602.04938' },
          { label: 'Captum: Lime', href: `${CAPTUM}lime.html` },
        ],
      },
      {
        id: 'KernelShap',
        title: 'Kernel SHAP',
        tags: ['Perturbation', 'Shapley'],
        summary: 'LIME with the SHAP kernel, so the surrogate weights approximate Shapley values.',
        differs: 'Same perturb-and-regress loop as LIME; only the sample weighting changes, chosen so the coefficients converge to Shapley values.',
        facts: [
          ['Family', 'Perturbation, Shapley'],
          ['Needs', 'A few hundred forward passes'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per segment'],
          ['Caveat', 'Same dependence on the segmentation as LIME'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: KernelShap', href: `${CAPTUM}kernel_shap.html` },
        ],
      },
      {
        id: 'ShapleyValueSampling',
        title: 'Shapley Value Sampling',
        tags: ['Perturbation', 'Shapley'],
        summary: 'Monte-Carlo Shapley values: samples orderings of the segments and measures each one\'s marginal contribution.',
        differs: 'The most literal of the Shapley estimators here, and by far the most expensive — no surrogate model in between.',
        facts: [
          ['Family', 'Perturbation, Shapley'],
          ['Needs', 'One pass per segment, per ordering'],
          ['Cost', 'Slowest'],
          ['Resolution', 'One value per segment'],
          ['Caveat', 'Cost grows with the segment count'],
        ],
        links: [
          { label: 'Captum: ShapleyValueSampling', href: `${CAPTUM}shapley_value_sampling.html` },
        ],
      },
      {
        id: 'FeatureAblation',
        title: 'Feature Ablation',
        tags: ['Perturbation'],
        summary: 'Replaces one feature group at a time with a baseline and measures the change in output.',
        differs: 'Occlusion without the sliding window: the groups are given explicitly, so they can follow segments rather than a grid.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One forward pass per group'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per group'],
          ['Caveat', 'A constant baseline creates inputs no model ever saw'],
        ],
        links: [
          { label: 'Captum: FeatureAblation', href: `${CAPTUM}feature_ablation.html` },
        ],
      },
      {
        id: 'FeaturePermutation',
        title: 'Feature Permutation',
        tags: ['Perturbation'],
        summary: 'Shuffles a feature group across the batch and measures the change in output.',
        differs: 'Keeps each feature\'s marginal distribution intact, which avoids the off-manifold inputs that ablating to a constant creates.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One pass per group, over a batch'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per group'],
          ['Caveat', 'Needs a batch to shuffle across'],
        ],
        links: [
          { label: 'Captum: FeaturePermutation', href: `${CAPTUM}feature_permutation.html` },
        ],
      },
      {
        id: '__segmentation__',
        title: 'SLIC / Quickshift / KMeans',
        tags: ['Segmentation'],
        summary: 'The suffix on a perturbation method names the algorithm used to cut the image into segments.',
        differs: 'SLIC grows compact superpixels on a grid, Quickshift follows density modes of colour and gives irregular regions, KMeans clusters pixels with no spatial constraint at all.',
        facts: [
          ['Family', 'Segmentation, not attribution'],
          ['Needs', 'One pass, before the method runs'],
          ['Cost', 'Negligible'],
          ['Resolution', 'It sets the segment size'],
          ['Caveat', 'Comparing one method under two segmentations shows how much of a map is the partition'],
        ],
        links: [
          { label: 'scikit-image: segmentation', href: 'https://scikit-image.org/docs/stable/api/skimage.segmentation.html' },
        ],
      },
    ],
  },
  {
    kind: 'metric',
    label: 'Metrics',
    entries: [
      {
        id: 'mif',
        title: 'MIF — Most Important First',
        tags: ['Faithfulness'],
        summary: 'Deletes pixels in descending order of attribution and tracks the class score, reported as the area under that curve.',
        differs: 'If the map is right, removing what it ranked highest destroys the prediction immediately — so the curve falls fast and the area is small.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Lower'],
          ['Removes', 'Pixels, most-attributed first'],
          ['Caveat', 'A map can game one order alone — read with LIF'],
        ],
        links: [],
      },
      {
        id: 'lif',
        title: 'LIF — Least Important First',
        tags: ['Faithfulness'],
        summary: 'The same deletion curve run in the opposite order: least-attributed pixels first.',
        differs: 'Removing what the map called irrelevant should leave the score alone, so here a faithful map keeps the curve high.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Higher'],
          ['Removes', 'Pixels, least-attributed first'],
          ['Caveat', 'Only meaningful read next to MIF'],
        ],
        links: [],
      },
      {
        id: 'morph',
        title: 'Morph — morphological faithfulness',
        tags: ['Faithfulness'],
        summary: 'A deletion curve whose regions are grown morphologically instead of pixel by pixel.',
        differs: 'Deleting scattered pixels produces inputs no model ever saw. Growing contiguous regions keeps the perturbed image plausible, at the cost of a coarser measurement.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Lower'],
          ['Removes', 'Morphologically grown regions'],
          ['Caveat', 'Coarser than the pixel-wise curves'],
        ],
        links: [],
      },
      {
        id: 'segment',
        title: 'Segment — segment-wise deletion',
        tags: ['Faithfulness'],
        summary: 'A deletion curve over segments rather than pixels.',
        differs: 'The unit of removal is a superpixel, which is what makes the score comparable between pixel-level and segment-level methods.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Lower'],
          ['Removes', 'Superpixels'],
          ['Caveat', 'Depends on the segmentation used'],
        ],
        links: [],
      },
      {
        id: 'infidelity',
        title: 'Infidelity',
        tags: ['Faithfulness'],
        summary: 'Mean squared gap between the change the attribution predicts and the change the model actually shows under random perturbations.',
        differs: 'The only one that does not delete anything: it asks the map to forecast the score move, then checks the forecast. Zero would mean a perfect local model.',
        facts: [
          ['Measures', 'Explanation error'],
          ['Better', 'Lower'],
          ['Removes', 'Nothing — it perturbs at random'],
          ['Caveat', 'Depends on the perturbation distribution'],
        ],
        links: [
          { label: 'Paper (arXiv 1901.09392)', href: 'https://arxiv.org/abs/1901.09392' },
        ],
      },
    ],
  },
];

const INDEX = new Map(
  WIKI_SECTIONS.flatMap((s) => s.entries.map((e) => [`${s.kind}:${e.id.toLowerCase()}`, e]))
);

const KIND_OF = new Map(
  WIKI_SECTIONS.flatMap((s) => s.entries.map((e) => [e, s.kind]))
);

// "Lime (SLIC)" -> "Lime". Segmentation variants share one entry.
function baseMethodId(id) {
  return String(id).replace(/\s*\([^)]*\)\s*$/, '').trim();
}

export function lookupWiki(kind, id) {
  if (!id) return null;
  const direct = INDEX.get(`${kind}:${String(id).toLowerCase()}`);
  if (direct) return direct;
  if (kind !== 'method') return null;
  return INDEX.get(`method:${baseMethodId(id).toLowerCase()}`) ?? null;
}

export function wikiKindOf(entry) {
  return KIND_OF.get(entry) ?? null;
}
