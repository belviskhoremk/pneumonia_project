// Front-end for the pneumonia detection page:
//  - previews the selected X-ray,
//  - sends it with the chosen models to POST /predict,
//  - displays the averaged probability and each model's own prediction.

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const imageInput = document.getElementById("image-upload");
    const modelSelect = document.getElementById("model-select");
    const uploadedImg = document.getElementById("uploaded-image");
    const resultDiv = document.getElementById("result");

    // Show a preview of the chosen image before it is uploaded.
    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];
        if (!file) return;
        uploadedImg.src = URL.createObjectURL(file);
        uploadedImg.style.display = "block";
    });

    /** Replace the result area with one paragraph per line of text. */
    function showResult(...lines) {
        resultDiv.replaceChildren(
            ...lines.map((text) => {
                const p = document.createElement("p");
                p.textContent = text; // textContent: never interpret as HTML
                return p;
            })
        );
    }

    const percent = (x) => `${(x * 100).toFixed(1)} %`;

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const selectedModels = Array.from(modelSelect.selectedOptions, (o) => o.value);
        if (imageInput.files.length === 0) return showResult("Please upload an image.");
        if (selectedModels.length === 0) return showResult("Please select at least one model.");

        const formData = new FormData();
        formData.append("image", imageInput.files[0]);
        formData.append("models", JSON.stringify(selectedModels));

        showResult("Analyzing...");
        try {
            const response = await fetch("/predict", { method: "POST", body: formData });
            const data = await response.json();
            if (!response.ok) return showResult(`Error: ${data.error}`);

            const names = Object.fromEntries(
                Array.from(modelSelect.options, (o) => [o.value, o.textContent])
            );
            showResult(
                `Pneumonia: ${data.pneumonia ? "Likely" : "Unlikely"}`,
                `Probability: ${percent(data.probability)}`,
                ...Object.entries(data.per_model).map(
                    ([id, p]) => `• ${names[id] ?? id}: ${percent(p)}`
                )
            );
        } catch (error) {
            console.error(error);
            showResult("An error occurred while processing your request.");
        }
    });
});
