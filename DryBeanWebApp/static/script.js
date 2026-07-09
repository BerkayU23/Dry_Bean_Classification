document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const fillRandomBtn = document.getElementById('fill-random');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const predictedClassSpan = document.getElementById('predicted-class');
    const resetBtn = document.getElementById('reset-btn');

    // List of input IDs to make it easier to gather data
    const featureIds = [
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", 
        "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter", 
        "Extent", "Solidity", "roundness", "Compactness", 
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
    ];

    // Typical ranges for a dry bean (approximate based on dataset stats) for random filling
    const randomRanges = {
        Area: [20000, 250000],
        Perimeter: [500, 2000],
        MajorAxisLength: [180, 750],
        MinorAxisLength: [120, 500],
        AspectRation: [1.0, 2.5],
        Eccentricity: [0.2, 0.95],
        ConvexArea: [20000, 250000],
        EquivDiameter: [150, 600],
        Extent: [0.5, 0.9],
        Solidity: [0.95, 0.99],
        roundness: [0.5, 0.99],
        Compactness: [0.5, 0.95],
        ShapeFactor1: [0.002, 0.01],
        ShapeFactor2: [0.0005, 0.004],
        ShapeFactor3: [0.3, 0.9],
        ShapeFactor4: [0.95, 0.999]
    };

    // Helper to generate random float between min and max
    function getRandomFloat(min, max) {
        return (Math.random() * (max - min) + min).toFixed(4);
    }

    // Fill Random button click handler
    fillRandomBtn.addEventListener('click', () => {
        featureIds.forEach(id => {
            const input = document.getElementById(id);
            if (input && randomRanges[id]) {
                input.value = getRandomFloat(randomRanges[id][0], randomRanges[id][1]);
            }
        });
        
        // Hide previous results/errors if any
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    });

    // Form submit handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Reset UI states
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        submitBtn.classList.add('btn-loading');

        // Gather data
        const payload = {};
        featureIds.forEach(id => {
            payload[id] = parseFloat(document.getElementById(id).value);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Bir hata oluştu.');
            }

            // Show Result
            predictedClassSpan.textContent = data.class;
            
            // Render probabilities
            const probsContainer = document.getElementById('probabilities-container');
            if (probsContainer && data.probabilities) {
                probsContainer.innerHTML = '<h4>Diğer Olasılıklar:</h4><br>';
                
                // Sort probabilities descending
                const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
                
                sortedProbs.forEach(([clsName, prob]) => {
                    const percentage = (prob * 100).toFixed(2);
                    
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'prob-item';
                    
                    itemDiv.innerHTML = `
                        <div class="prob-label">
                            <span>${clsName}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${percentage}%"></div>
                        </div>
                    `;
                    probsContainer.appendChild(itemDiv);
                });
            }

            form.style.display = 'none';
            resultContainer.classList.remove('hidden');
            
        } catch (error) {
            console.error('Prediction Error:', error);
            errorContainer.textContent = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            submitBtn.classList.remove('btn-loading');
        }
    });

    // Reset button click handler
    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        form.style.display = 'block';
        // optionally clear form: form.reset();
    });

    // Mouse un gittiği yöne doğru arka plan renk değişimi
    document.addEventListener('mousemove', (e) => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        // HSL değerleriyle yumuşak ve şık bir renk geçişi (Mavi - Mor arası)
        const hue1 = Math.round(210 + (x * 50)); 
        const hue2 = Math.round(250 + (y * 50)); 
        
        document.body.style.background = `radial-gradient(circle at ${e.clientX}px ${e.clientY}px, hsl(${hue1}, 60%, 15%), hsl(${hue2}, 70%, 8%))`;
    });
});
