document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('result-content');

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const data = {
            model: form.model.value,
            sepal_length: form.sepal_length.value,
            sepal_width: form.sepal_width.value,
            petal_length: form.petal_length.value,
            petal_width: form.petal_width.value
        };

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(json => {
            resultDiv.style.display = 'block';
            if (json.success) {
                if (json.prediction !== undefined) {
                    resultContent.innerHTML = `<strong>Prediction:</strong> ${json.prediction}<br>`;
                    if (json.probabilities) {
                        resultContent.innerHTML += `<strong>Probabilities:</strong> ${JSON.stringify(json.probabilities)}`;
                    }
                } else if (json.result) {
                    resultContent.innerHTML = `<strong>Transformed Data:</strong> ${JSON.stringify(json.result)}`;
                } else {
                    resultContent.innerHTML = `<strong>Message:</strong> ${json.message}`;
                }
            } else {
                resultContent.innerHTML = `<span class="text-danger">Error: ${json.message}</span>`;
            }
        })
        .catch(err => {
            resultDiv.style.display = 'block';
            resultContent.innerHTML = `<span class="text-danger">Error: ${err}</span>`;
        });
    });
});