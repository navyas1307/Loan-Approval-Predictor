document.getElementById("loan-form").addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent the default form submission

    const resultDiv = document.getElementById("result");
    const limeMessageDiv = document.getElementById("lime-message");
    const limeExplanationDiv = document.getElementById("lime-explanation");
    resultDiv.innerHTML = "Predicting...";
    resultDiv.className = "";
    limeMessageDiv.style.display = "none";
    limeExplanationDiv.style.display = "none";

    try {
        // Collect form data
        const formData = {
            income: document.getElementById("income").value,
            cibil: document.getElementById("cibil").value,
            loan_amount: document.getElementById("loan_amount").value,
            loan_term: document.getElementById("loan_term").value,
            residential_assets_value: document.getElementById("residential_assets").value,
            commercial_assets_value: document.getElementById("commercial_assets").value,
            luxury_assets_value: document.getElementById("luxury_assets").value,
            gender: document.getElementById("gender").value,
            married: document.getElementById("married").value,
            dependents: document.getElementById("dependents").value,
            education: document.getElementById("education").value,
            self_employed: document.getElementById("self_employed").value
        };

        // Make the fetch request to the server
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formData)
        });

        // Check if the response is successful
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }

        // Parse the response as JSON
        const result = await response.json();
        console.log(result); // Log the result for debugging

        // Display the result message
        resultDiv.textContent = result.message;

        // Update the UI based on the result
        if (result.message.includes("Approved")) {
            resultDiv.classList.add("approved");
        } else {
            resultDiv.classList.add("rejected");
            limeMessageDiv.style.display = "block";
            limeMessageDiv.textContent = result.lime_message || "No additional explanation provided.";
            if (result.lime_explanation && Array.isArray(result.lime_explanation)) {
                limeExplanationDiv.style.display = "block";
                limeExplanationDiv.innerHTML = `
                    <h3>Explanation of LIME Message:</h3>
                    <ul>
                        ${result.lime_explanation.map(([feature, importance]) => `
                            <li>${feature}: ${importance.toFixed(2)} (${importance > 0 ? 'Positive' : 'Negative'} contribution)</li>
                        `).join('')}
                    </ul>
                    <p>Positive values indicate features that contribute positively towards loan approval, while negative values indicate features that contribute negatively.</p>
                `;
            } else {
                limeExplanationDiv.innerHTML = "<p>No detailed LIME explanation available.</p>";
            }
        }
    } catch (error) {
        console.error("Error:", error);
        resultDiv.textContent = "An error occurred while predicting loan status.";
        resultDiv.classList.add("rejected");
    }
});