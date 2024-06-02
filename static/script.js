document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("upload-form");
    const imageInput = document.getElementById("image-upload");
    const uploadedImg = document.getElementById("uploaded-image");
    const resultDiv = document.getElementById("result");

    imageInput.addEventListener("change", function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImg.src = e.target.result;
                uploadedImg.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });

    form.addEventListener("submit", function(event) {
        event.preventDefault();

        const selectedModels = Array.from(document.getElementById("model-select").selectedOptions).map(option => option.value);
        if (imageInput.files.length === 0) {
            resultDiv.textContent = "Please upload an image.";
            return;
        }

        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append("image", file);
        formData.append("models", JSON.stringify(selectedModels));

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = `
                <p>Probability: ${data.probability}</p>
                <p>Pneumonia: ${data.pneumonia ? "Yes" : "No"}</p>
            `;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.textContent = "An error occurred while processing your request.";
        });
    });
});
