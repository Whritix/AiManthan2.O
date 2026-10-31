# Unified defect detector: scope and class taxonomy

## Scope

The first unified detector will cover the five product families already present in the application: steel, metal nut, screw, transistor, and bottle. It will predict 21 product-specific defect classes. It is a closed-set model: products and defect types outside this list must be treated as unknown and routed for review.

The canonical class IDs and names are in [`data/defect_taxonomy.json`](../data/defect_taxonomy.json). Keep those IDs stable when creating labels and the YOLO dataset YAML.

## Classes

| ID | YOLO class | Product profile | Existing profile class |
|---:|---|---|---|
| 0 | `steel_crazing` | Steel | `crazing` |
| 1 | `steel_inclusion` | Steel | `inclusion` |
| 2 | `steel_patches` | Steel | `patches` |
| 3 | `steel_pitted_surface` | Steel | `pitted_surface` |
| 4 | `steel_scratches` | Steel | `scratches` |
| 5 | `metal_nut_bent` | Metal nut | `bent` |
| 6 | `metal_nut_color` | Metal nut | `color` |
| 7 | `metal_nut_flip` | Metal nut | `flip` |
| 8 | `metal_nut_scratch` | Metal nut | `scratch` |
| 9 | `screw_manipulated_front` | Screw | `manipulated_front` |
| 10 | `screw_scratch_head` | Screw | `scratch_head` |
| 11 | `screw_scratch_neck` | Screw | `scratch_neck` |
| 12 | `screw_thread_side` | Screw | `thread_side` |
| 13 | `screw_thread_top` | Screw | `thread_top` |
| 14 | `transistor_bent_lead` | Transistor | `bent_lead` |
| 15 | `transistor_cut_lead` | Transistor | `cut_lead` |
| 16 | `transistor_damaged_case` | Transistor | `damaged_case` |
| 17 | `transistor_misplaced` | Transistor | `misplaced` |
| 18 | `bottle_broken_large` | Bottle | `broken_large` |
| 19 | `bottle_broken_small` | Bottle | `broken_small` |
| 20 | `bottle_contamination` | Bottle | `contamination` |

## Output meaning

The display percentage for a detection is the model's confidence in that class and box. It is not the fraction of the product's surface that is defective. Measuring affected surface area requires segmentation masks and a segmentation model.

The defect name must remain product-specific. For example, `metal_nut_scratch` and `screw_scratch_head` are separate classes even though both contain the word “scratch.”
