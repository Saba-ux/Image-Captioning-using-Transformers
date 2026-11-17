import pickle
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

results_path = "test_results_colab/0310.1454/selected.pickle"
images_path = "data/mscoco_h5/test_images.hdf5"
output_dir = Path("qualitative_results")
output_dir.mkdir(exist_ok=True)

with open(results_path, "rb") as f:
    results = pickle.load(f)

h5_file = h5py.File(images_path, "r")
images = h5_file["test"]

def format_caption(tokens):
    caption = " ".join(tokens)
    return caption.replace(" .", ".").replace(" ,", ",")

metrics = ["bleu1", "bleu2", "bleu3", "bleu4", "gleu", "meteor"]
pdf_path = output_dir / "qualitative_evaluation.pdf"

with PdfPages(pdf_path) as pdf:
    for i in range(10):
        image = images[i].transpose(1, 2, 0)
        prediction = format_caption(results["hypos_text"].iloc[i])
        references = results["refs_text"].iloc[i]

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(
            f"Qualitative Evaluation - Test Image {i + 1}",
            fontsize=16,
            fontweight="bold"
        )

        ax_img = fig.add_axes([0.05, 0.20, 0.45, 0.675])
        ax_img.set_box_aspect(1)
        
        ax_img.imshow(image, aspect="equal")
        ax_img.axis("off")

        ax_text = fig.add_axes([0.53, 0.08, 0.43, 0.82])
        ax_text.axis("off")

        text = "GENERATED CAPTION\n"
        text += prediction + "\n\n"
        text += "HUMAN REFERENCE CAPTIONS\n"

        for j, ref in enumerate(references, 1):
            text += f"{j}. {format_caption(ref)}\n"

        text += "\nEVALUATION SCORES\n"

        for metric in metrics:
            score = results[metric].iloc[i]
            text += f"{metric.upper()}: {score:.4f}\n"

        ax_text.text(
            0, 1, text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment="top",
            linespacing=1.5,
            wrap=True
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Processed image {i + 1}/10")

h5_file.close()
print(f"\nPDF saved at: {pdf_path}")
