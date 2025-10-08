// MarketToM Frontend Application Logic

const API_BASE = '';  // Flask runs on the same domain

// Global state
let currentDatasets = [];
let currentStocks = [];
let isRunning = false;

// DOM elements
const datasetSelect = document.getElementById('dataset-select');
const splitSelect = document.getElementById('split-select');
const stockSelect = document.getElementById('stock-select');
const dayIndexInput = document.getElementById('day-index');
const windowSizeInput = document.getElementById('window-size');
const runBtn = document.getElementById('run-inference-btn');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const welcomeScreen = document.getElementById('welcome-screen');
const resultsScreen = document.getElementById('results-screen');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    datasetSelect.addEventListener('change', onDatasetChange);
    splitSelect.addEventListener('change', onSplitChange);
    stockSelect.addEventListener('change', onStockChange);
    runBtn.addEventListener('click', runInference);
    
    // Modal window
    const modal = document.getElementById('modal');
    const closeBtn = document.getElementsByClassName('close')[0];
    closeBtn.onclick = () => modal.style.display = 'none';
    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
}

// Load dataset list
async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/api/datasets`);
        const data = await response.json();
        
        if (data.datasets) {
            currentDatasets = data.datasets;
            populateDatasetSelect(data.datasets);
        }
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showError('Failed to load datasets: ' + error.message);
    }
}

// Populate dataset dropdown
function populateDatasetSelect(datasets) {
    datasetSelect.innerHTML = '<option value="">Select dataset</option>';
    datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.name;
        option.textContent = dataset.name;
        option.dataset.splits = JSON.stringify(dataset.splits);
        datasetSelect.appendChild(option);
    });
}

// Dataset selection changed
function onDatasetChange() {
    const selectedOption = datasetSelect.options[datasetSelect.selectedIndex];
    if (!selectedOption.value) {
        splitSelect.innerHTML = '<option value="">Select dataset first</option>';
        stockSelect.innerHTML = '<option value="">Select split first</option>';
        runBtn.disabled = true;
        return;
    }
    
    const splits = JSON.parse(selectedOption.dataset.splits || '[]');
    populateSplitSelect(splits);
}

// Populate split dropdown
function populateSplitSelect(splits) {
    splitSelect.innerHTML = '<option value="">Select split</option>';
    splits.forEach(split => {
        const option = document.createElement('option');
        option.value = split;
        option.textContent = split;
        splitSelect.appendChild(option);
    });
    stockSelect.innerHTML = '<option value="">Select split first</option>';
    runBtn.disabled = true;
}

// Split selection changed
async function onSplitChange() {
    const dataset = datasetSelect.value;
    const split = splitSelect.value;
    
    if (!dataset || !split) {
        stockSelect.innerHTML = '<option value="">Select split first</option>';
        runBtn.disabled = true;
        return;
    }
    
    stockSelect.innerHTML = '<option value="">Loading...</option>';
    stockSelect.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/stocks/${dataset}/${split}`);
        const data = await response.json();
        
        if (data.stocks) {
            currentStocks = data.stocks;
            populateStockSelect(data.stocks);
        }
    } catch (error) {
        console.error('Failed to load stock list:', error);
        stockSelect.innerHTML = '<option value="">Loading failed</option>';
    } finally {
        stockSelect.disabled = false;
    }
}

// Populate stock dropdown
function populateStockSelect(stocks) {
    stockSelect.innerHTML = '<option value="">Select stock</option>';
    stocks.forEach(stock => {
        const option = document.createElement('option');
        option.value = stock;
        option.textContent = stock;
        stockSelect.appendChild(option);
    });
    runBtn.disabled = true;
}

// Stock selection changed
function onStockChange() {
    const stock = stockSelect.value;
    runBtn.disabled = !stock;
}

// Run inference
async function runInference() {
    if (isRunning) return;
    
    const dataset = datasetSelect.value;
    const split = splitSelect.value;
    const stock = stockSelect.value;
    const dayIndex = parseInt(dayIndexInput.value) || 0;
    const windowSize = parseInt(windowSizeInput.value) || 5;
    
    if (!dataset || !split || !stock) {
        showError('Please select dataset, split, and stock');
        return;
    }
    
    isRunning = true;
    runBtn.disabled = true;
    progressContainer.style.display = 'block';
    welcomeScreen.style.display = 'none';
    resultsScreen.style.display = 'none';
    
    // Show results area immediately with "waiting" content
    welcomeScreen.style.display = 'none';
    resultsScreen.style.display = 'block';
    
    // Set initial placeholder text
    document.getElementById('belief-content').innerHTML = '<em style="color: #94a3b8;">⏳ Analyzing market belief...</em>';
    document.getElementById('intent-content').innerHTML = '<em style="color: #94a3b8;">⏳ Waiting for belief analysis...</em>';
    document.getElementById('emotion-content').innerHTML = '<em style="color: #94a3b8;">⏳ Waiting for belief analysis...</em>';
    document.getElementById('predicted-action').textContent = '⏳';
    document.getElementById('actual-action').textContent = 'Loading...';
    document.getElementById('confidence-score').textContent = '...';
    document.getElementById('accuracy-badge').textContent = 'Waiting...';
    
    // Start polling status
    const statusInterval = setInterval(async () => {
        const status = await pollInferenceStatus();
        if (status === 'completed' || status === 'error') {
            clearInterval(statusInterval);
            isRunning = false;
            runBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
    }, 500);
    
    try {
        // Send inference request (returns immediately, does not wait for completion)
        const response = await fetch(`${API_BASE}/api/run_inference`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset,
                split,
                stock,
                day_index: dayIndex,
                window_size: windowSize
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            clearInterval(statusInterval);
            showError('Failed to start inference: ' + (data.error || 'Unknown error'));
            isRunning = false;
            runBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
        
    } catch (error) {
        clearInterval(statusInterval);
        console.error('Inference request failed:', error);
        showError('Inference request failed: ' + error.message);
        isRunning = false;
        runBtn.disabled = false;
        progressContainer.style.display = 'none';
    }
}

// Extract mental state description from JSON or return as is
function extractMentalStateDescription(state) {
    if (!state) return 'No data';
    if (typeof state === 'string') {
        try {
            // Try to parse as JSON
            const parsed = JSON.parse(state);
            if (parsed['mental state description']) {
                return parsed['mental state description'];
            }
        } catch (e) {
            // Not JSON, return as is
        }
    }
    return state;
}

// Display results
function displayResults(results) {
    console.log('Results:', results);
    
    // Show results screen
    welcomeScreen.style.display = 'none';
    resultsScreen.style.display = 'block';
    
    // Basic information
    document.getElementById('result-dataset').textContent = results.dataset;
    document.getElementById('result-stock').textContent = results.stock;
    document.getElementById('result-day').textContent = results.day;
    document.getElementById('result-timestamp').textContent = 
        new Date(results.timestamp).toLocaleString('en-US');
    
    // Prediction results
    const predictedAction = results.action_prediction.predicted_action;
    const actualLabel = results.actual_label;
    const isCorrect = results.is_correct;
    const confidence = (results.action_prediction.confidence * 100).toFixed(2);
    
    const predictedBadge = document.getElementById('predicted-action');
    predictedBadge.textContent = predictedAction;
    predictedBadge.className = 'action-badge ' + predictedAction.toLowerCase();
    
    const actualBadge = document.getElementById('actual-action');
    actualBadge.textContent = actualLabel;
    actualBadge.className = 'action-badge ' + actualLabel.toLowerCase();
    
    const accuracyBadge = document.getElementById('accuracy-badge');
    accuracyBadge.textContent = isCorrect ? '✓ Correct' : '✗ Incorrect';
    accuracyBadge.className = 'accuracy-badge ' + (isCorrect ? 'correct' : 'incorrect');
    
    document.getElementById('confidence-score').textContent = confidence + '%';
    
    // Mental states - extract description from JSON if needed
    document.getElementById('belief-content').textContent = extractMentalStateDescription(results.mental_states.belief);
    document.getElementById('intent-content').textContent = extractMentalStateDescription(results.mental_states.intent);
    document.getElementById('emotion-content').textContent = extractMentalStateDescription(results.mental_states.emotion);
    
    // Retrieved strategies
    if (results.retrieved_strategies && Object.keys(results.retrieved_strategies).length > 0) {
        document.getElementById('strategies-card').style.display = 'block';
        displayStrategies(results.retrieved_strategies);
    } else {
        document.getElementById('strategies-card').style.display = 'none';
    }
    
    // Backward inference
    if (results.backward_inference) {
        document.getElementById('backward-card').style.display = 'block';
        displayBackwardInference(results.backward_inference);
    } else {
        document.getElementById('backward-card').style.display = 'none';
    }
    
    // Environment data
    displayEnvironmentData(results.environment_state);
    
    // Visualization - will be displayed when available via intermediate results
    // (Already handled in displayIntermediateResults)
}

// Display retrieved strategies
function displayStrategies(strategies) {
    const container = document.getElementById('retrieved-strategies-content');
    container.innerHTML = '';
    
    for (const [type, strategyList] of Object.entries(strategies)) {
        if (strategyList && strategyList.length > 0) {
            strategyList.forEach((strategy, index) => {
                const div = document.createElement('div');
                div.className = 'strategy-item';
                div.innerHTML = `
                    <h5>${type.toUpperCase()} Strategy #${index + 1}</h5>
                    <p>${strategy.content || JSON.stringify(strategy)}</p>
                `;
                container.appendChild(div);
            });
        }
    }
}

// Display backward inference results
function displayBackwardInference(backward) {
    const container = document.getElementById('backward-content');
    container.innerHTML = '';
    
    if (backward.updates) {
        backward.updates.forEach((update, index) => {
            const div = document.createElement('div');
            div.className = 'backward-action';
            div.innerHTML = `
                <h5>Update #${index + 1}: ${update.action || 'N/A'}</h5>
                <p><strong>Type:</strong> ${update.state_type || 'N/A'}</p>
                <p><strong>Content:</strong> ${update.content || update.strategy || 'N/A'}</p>
            `;
            container.appendChild(div);
        });
    } else {
        container.innerHTML = '<p>No strategy updates</p>';
    }
}

// Display environment data
function displayEnvironmentData(envState) {
    // Text samples
    const textsContainer = document.getElementById('sample-texts');
    textsContainer.innerHTML = '';
    
    if (envState.sample_texts && envState.sample_texts.length > 0) {
        envState.sample_texts.forEach((text, index) => {
            const div = document.createElement('div');
            div.className = 'text-sample';
            div.textContent = `${index + 1}. ${text}`;
            textsContainer.appendChild(div);
        });
    } else {
        textsContainer.innerHTML = '<p>No text data</p>';
    }
    
    // Price data
    const priceContainer = document.getElementById('price-data');
    priceContainer.innerHTML = '';
    
    if (envState.price_data && Object.keys(envState.price_data).length > 0) {
        for (const [key, value] of Object.entries(envState.price_data)) {
            const div = document.createElement('div');
            div.className = 'price-item';
            div.innerHTML = `
                <div class="price-label">${key}</div>
                <div class="price-value">${value}</div>
            `;
            priceContainer.appendChild(div);
        }
    } else {
        priceContainer.innerHTML = '<p>No price data</p>';
    }
}

// View strategies
async function viewStrategies() {
    try {
        const response = await fetch(`${API_BASE}/api/strategies`);
        const data = await response.json();
        
        if (data.strategies) {
            showModal('Strategy Database', formatStrategies(data.strategies));
        }
    } catch (error) {
        console.error('Failed to load strategies:', error);
        showError('Failed to load strategies: ' + error.message);
    }
}

// Format strategies
function formatStrategies(strategies) {
    let html = '';
    for (const [type, strategyList] of Object.entries(strategies)) {
        html += `<h3>${type.toUpperCase()} Strategies (${strategyList.length})</h3>`;
        if (strategyList.length > 0) {
            html += '<ul style="list-style: decimal; padding-left: 20px;">';
            strategyList.slice(0, 10).forEach(strategy => {
                html += `<li style="margin-bottom: 10px;">${JSON.stringify(strategy, null, 2)}</li>`;
            });
            if (strategyList.length > 10) {
                html += `<li><em>${strategyList.length - 10} more...</em></li>`;
            }
            html += '</ul>';
        } else {
            html += '<p>No strategies yet</p>';
        }
    }
    return html;
}

// View inference logs
async function viewLogs() {
    try {
        const response = await fetch(`${API_BASE}/api/inference_logs`);
        const data = await response.json();
        
        if (data.logs) {
            showModal('Inference Logs', formatLogs(data.logs));
        }
    } catch (error) {
        console.error('Failed to load logs:', error);
        showError('Failed to load logs: ' + error.message);
    }
}

// Format logs
function formatLogs(logs) {
    if (logs.length === 0) {
        return '<p>No inference history yet</p>';
    }
    
    let html = '<table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr style="background: #f1f5f9;">';
    html += '<th style="padding: 10px; text-align: left;">Filename</th>';
    html += '<th style="padding: 10px; text-align: left;">Time</th>';
    html += '<th style="padding: 10px; text-align: right;">Size</th>';
    html += '</tr></thead><tbody>';
    
    logs.forEach(log => {
        html += '<tr style="border-bottom: 1px solid #e2e8f0;">';
        html += `<td style="padding: 10px;">${log.filename}</td>`;
        html += `<td style="padding: 10px;">${new Date(log.timestamp).toLocaleString('en-US')}</td>`;
        html += `<td style="padding: 10px; text-align: right;">${(log.size / 1024).toFixed(2)} KB</td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

// View system configuration
async function viewConfig() {
    try {
        const response = await fetch(`${API_BASE}/api/config`);
        const data = await response.json();
        
        if (data.config) {
            showModal('System Configuration', `<pre style="background: #f1f5f9; padding: 15px; border-radius: 6px; overflow-x: auto;">${JSON.stringify(data.config, null, 2)}</pre>`);
        }
    } catch (error) {
        console.error('Failed to load configuration:', error);
        showError('Failed to load configuration: ' + error.message);
    }
}

// Show modal window
function showModal(title, content) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = content;
    document.getElementById('modal').style.display = 'block';
}

// Show error
function showError(message) {
    alert('Error: ' + message);
}

// Poll inference status
async function pollInferenceStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/inference_status`);
        const data = await response.json();
        
        // Update progress bar and text
        if (data.progress !== undefined) {
            progressFill.style.width = data.progress + '%';
            progressText.textContent = data.current_step || 'Processing...';
        }
        
        // Display intermediate results in real-time
        if (data.intermediate_results) {
            console.log('📥 Received intermediate results, progress:', data.progress, '%');
            displayIntermediateResults(data.intermediate_results);
        }
        
        // If inference completed, display results
        if (data.status === 'completed' && data.results) {
            displayResults(data.results);
            return 'completed';
        }
        
        // If there's an error, show error
        if (data.status === 'error' && data.error) {
            console.error('Inference error:', data.error);
            if (data.traceback) {
                console.error('Error details:', data.traceback);
            }
            showError('Inference failed: ' + data.error);
            return 'error';
        }
        
        return data.status || 'running';
    } catch (error) {
        console.error('Failed to poll status:', error);
        return 'error';
    }
}

// Display intermediate results (real-time updates)
function displayIntermediateResults(intermediate) {
    console.log('🔄 Updating intermediate results:', intermediate);
    
    // Display environment data
    if (intermediate.environment) {
        console.log('✅ Updating environment data:', intermediate.environment);
        const env = intermediate.environment;
        
        // Update text samples
        const textsContainer = document.getElementById('sample-texts');
        if (env.sample_texts && env.sample_texts.length > 0) {
            textsContainer.innerHTML = '';
            env.sample_texts.forEach((text, index) => {
                const div = document.createElement('div');
                div.className = 'text-sample';
                div.textContent = `${index + 1}. ${text}`;
                textsContainer.appendChild(div);
            });
        }
        
        // Update price data
        const priceContainer = document.getElementById('price-data');
        console.log('📊 Checking price data:', env.price_data, 'keys:', env.price_data ? Object.keys(env.price_data).length : 0);
        
        if (env.price_data && Object.keys(env.price_data).length > 0) {
            priceContainer.innerHTML = '';
            for (const [key, value] of Object.entries(env.price_data)) {
                console.log('📈 Adding price item:', key, '=', value);
                const div = document.createElement('div');
                div.className = 'price-item';
                div.innerHTML = `
                    <div class="price-label">${key}</div>
                    <div class="price-value">${value}</div>
                `;
                priceContainer.appendChild(div);
            }
        } else {
            priceContainer.innerHTML = '<em style="color: #f59e0b;">⚠️ No price data for this date</em>';
            console.log('⚠️ No price data');
        }
    }
    
    // Update mental states (if available)
    if (intermediate.belief && intermediate.belief !== 'N/A') {
        const now = new Date().toLocaleTimeString('en-US');
        const backendTime = intermediate.belief_time || 'unknown';
        const beliefText = extractMentalStateDescription(intermediate.belief);
        console.log(`✅ [${now}] Frontend received belief (backend generated at: ${backendTime}):`, beliefText.substring(0, 50) + '...');
        
        const beliefEl = document.getElementById('belief-content');
        beliefEl.innerHTML = `
            <strong style="color: #10b981;">✓ Completed</strong>
            <small style="color: #64748b;"> (backend: ${backendTime}, frontend received: ${now})</small>
            <br>${beliefText}
        `;
        
        // Update hint text
        if (!intermediate.intent || intermediate.intent === 'N/A') {
            document.getElementById('intent-content').innerHTML = '<em style="color: #f59e0b;">⏳ Inferring market intent...</em>';
        }
    }
    
    if (intermediate.intent && intermediate.intent !== 'N/A') {
        const now = new Date().toLocaleTimeString('en-US');
        const backendTime = intermediate.intent_time || 'unknown';
        const intentText = extractMentalStateDescription(intermediate.intent);
        console.log(`✅ [${now}] Frontend received intent (backend generated at: ${backendTime}):`, intentText.substring(0, 50) + '...');
        
        const intentEl = document.getElementById('intent-content');
        intentEl.innerHTML = `
            <strong style="color: #10b981;">✓ Completed</strong>
            <small style="color: #64748b;"> (backend: ${backendTime}, frontend received: ${now})</small>
            <br>${intentText}
        `;
        
        // Update hint text
        if (!intermediate.emotion || intermediate.emotion === 'N/A') {
            document.getElementById('emotion-content').innerHTML = '<em style="color: #f59e0b;">⏳ Analyzing market emotion...</em>';
        }
    }
    
    if (intermediate.emotion && intermediate.emotion !== 'N/A') {
        const now = new Date().toLocaleTimeString('en-US');
        const backendTime = intermediate.emotion_time || 'unknown';
        const emotionText = extractMentalStateDescription(intermediate.emotion);
        console.log(`✅ [${now}] Frontend received emotion (backend generated at: ${backendTime}):`, emotionText.substring(0, 50) + '...');
        
        const emotionEl = document.getElementById('emotion-content');
        emotionEl.innerHTML = `
            <strong style="color: #10b981;">✓ Completed</strong>
            <small style="color: #64748b;"> (backend: ${backendTime}, frontend received: ${now})</small>
            <br>${emotionText}
        `;
    }
    
    // Display prediction (if available)
    if (intermediate.predicted_action) {
        console.log('✅ Updating prediction:', intermediate.predicted_action, 'confidence:', intermediate.confidence);
        
        const predictedBadge = document.getElementById('predicted-action');
        predictedBadge.textContent = intermediate.predicted_action;
        predictedBadge.className = 'action-badge ' + intermediate.predicted_action.toLowerCase();
        
        if (intermediate.confidence) {
            document.getElementById('confidence-score').textContent = 
                (intermediate.confidence * 100).toFixed(2) + '%';
        }
        
        // Display expert judgments
        if (intermediate.expert_samples && intermediate.expert_samples.length > 0) {
            console.log('📊 Received expert judgments:', intermediate.expert_samples.length);
            
            // Show expert judgments card
            const expertCard = document.getElementById('expert-samples-card');
            const expertContent = document.getElementById('expert-samples-content');
            expertCard.style.display = 'block';
            expertContent.innerHTML = '';
            
            intermediate.expert_samples.forEach((sample, idx) => {
                console.log(`  Expert ${sample.index} (${sample.role ? sample.role.substring(0, 30) : 'unknown'}): up probability ${(sample.probability * 100).toFixed(1)}%`);
                
                // Create expert judgment card
                const probPercent = (sample.probability * 100).toFixed(1);
                const isHigh = sample.probability > 0.5;
                
                // Extract expert role name (first sentence or first 50 chars)
                let roleName = 'Expert ' + sample.index;
                if (sample.role) {
                    const roleShort = sample.role.split('.')[0].substring(0, 60);
                    roleName = roleShort;
                }
                
                const expertDiv = document.createElement('div');
                expertDiv.className = 'expert-sample-item';
                expertDiv.innerHTML = `
                    <div class="expert-sample-header">
                        <span class="expert-number">🧠 Expert ${sample.index}</span>
                        <span class="expert-probability ${isHigh ? 'high' : 'low'}">${isHigh ? '📈' : '📉'} ${probPercent}%</span>
                    </div>
                    <div class="expert-role" style="font-size: 0.9rem; color: #78350f; margin-bottom: 8px; line-height: 1.4;">
                        <strong>Role:</strong> ${roleName}
                    </div>
                    ${sample.reasoning ? `
                        <div class="expert-reasoning" style="font-size: 0.85rem; color: #92400e; margin-bottom: 8px; line-height: 1.5; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 4px;">
                            <strong>Analysis:</strong> ${sample.reasoning}
                        </div>
                    ` : ''}
                    <div class="expert-meta">
                        <div class="expert-meta-item">
                            <strong>Confidence:</strong> ${sample.log_confidence.toFixed(3)}
                        </div>
                        <div class="expert-meta-item">
                            <strong>Weight:</strong> ${(sample.normalized_weight * 100).toFixed(1)}%
                        </div>
                    </div>
                `;
                expertContent.appendChild(expertDiv);
            });
        }
    }
    
    // Display actual label and accuracy
    if (intermediate.actual_label) {
        console.log('✅ Updating actual label:', intermediate.actual_label);
        const actualBadge = document.getElementById('actual-action');
        actualBadge.textContent = intermediate.actual_label;
        actualBadge.className = 'action-badge ' + intermediate.actual_label.toLowerCase();
    }
    
    if (intermediate.is_correct !== undefined) {
        const accuracyBadge = document.getElementById('accuracy-badge');
        accuracyBadge.textContent = intermediate.is_correct ? '✓ Correct' : '✗ Incorrect';
        accuracyBadge.className = 'accuracy-badge ' + (intermediate.is_correct ? 'correct' : 'incorrect');
    }
    
    // Display backward inference results
    console.log('🔍 Checking for backward_result in intermediate:', 'backward_result' in intermediate, intermediate.backward_result);
    
    if (intermediate.backward_result) {
        console.log('✅ Updating backward inference results:', intermediate.backward_result);
        console.log('📊 Type:', typeof intermediate.backward_result);
        console.log('📊 Keys:', Object.keys(intermediate.backward_result));
        
        const backwardCard = document.getElementById('backward-card');
        const backwardContent = document.getElementById('backward-content');
        backwardCard.style.display = 'block';
        
        // Parse backward inference results
        let resultHTML = '';
        
        if (typeof intermediate.backward_result === 'object') {
            // If it's an object, display strategy database updates
            const updates = intermediate.backward_result['strategy_updates'] || intermediate.backward_result || {};
            
            console.log('📋 Strategy database update content:', updates);
            console.log('📋 Update keys:', Object.keys(updates));
            console.log('📋 Update entries:', Object.entries(updates));
            
            if (Object.keys(updates).length > 0) {
                resultHTML += '<div style="margin-top: 12px;">';
                
                // Level name mapping
                const levelNames = {
                    'belief': 'Belief Strategy',
                    'intent': 'Intent Strategy',
                    'emotion': 'Emotion Strategy'
                };
                
                for (const [level, strategies] of Object.entries(updates)) {
                    const levelName = levelNames[level] || level;
                    resultHTML += `
                        <div style="margin-bottom: 16px; padding: 12px; background: rgba(59, 130, 246, 0.05); border-left: 3px solid #3b82f6; border-radius: 4px;">
                            <h4 style="margin: 0 0 12px 0; color: #1e40af; font-size: 1rem;">📚 ${levelName}</h4>
                    `;
                    
                    if (Array.isArray(strategies)) {
                        strategies.forEach((strategy, idx) => {
                            const typeColor = strategy.type === 'CREATE' || strategy.type === 'create' ? '#10b981' : '#f59e0b';
                            const typeIcon = strategy.type === 'CREATE' || strategy.type === 'create' ? '✨' : '🔄';
                            const typeText = strategy.type === 'CREATE' || strategy.type === 'create' ? 'CREATE' : strategy.type === 'MODIFY' || strategy.type === 'modify' ? 'MODIFY' : strategy.type;
                            
                            resultHTML += `
                                <div style="margin-bottom: 12px; padding: 10px; background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                        <span style="color: ${typeColor}; font-weight: 600; font-size: 0.9rem;">${typeIcon} ${typeText}</span>
                                        <span style="color: #94a3b8; font-size: 0.85rem; font-family: monospace;">${strategy.id || 'N/A'}</span>
                                    </div>
                                    <div style="color: #1e293b; line-height: 1.5; font-size: 0.9rem;">
                                        ${strategy.content || 'No content'}
                                    </div>
                                </div>
                            `;
                        });
                    }
                    
                    resultHTML += '</div>';
                }
                resultHTML += '</div>';
            } else {
                resultHTML = '<p style="color: #64748b; padding: 12px;">✓ Strategy database not updated (prediction may be correct or no adjustment needed)</p>';
            }
        } else {
            // If it's a string, display directly
            resultHTML = `<p style="color: #1e293b; line-height: 1.6; padding: 12px;">${intermediate.backward_result}</p>`;
        }
        
        backwardContent.innerHTML = resultHTML;
    }
    
    // Display visualization if available
    if (intermediate.visualization) {
        console.log('🎨 Visualization available:', intermediate.visualization);
        displayVisualization(intermediate.visualization);
    }
}

// Display visualization
function displayVisualization(filename) {
    const vizCard = document.getElementById('visualization-card');
    const vizLoading = document.getElementById('visualization-loading');
    const vizImage = document.getElementById('visualization-image');
    const vizError = document.getElementById('visualization-error');
    const vizImg = document.getElementById('visualization-img');
    
    // Show visualization card
    vizCard.style.display = 'block';
    
    // Hide error if any
    vizError.style.display = 'none';
    
    // Show loading
    vizLoading.style.display = 'block';
    vizImage.style.display = 'none';
    
    // Load image with cache-busting timestamp
    const timestamp = new Date().getTime();
    const imgUrl = `/visualizations/${filename}?t=${timestamp}`;
    
    // Create a new image to test loading
    const testImg = new Image();
    testImg.onload = function() {
        console.log('✅ Visualization image loaded successfully');
        vizImg.src = imgUrl;
        vizLoading.style.display = 'none';
        vizImage.style.display = 'block';
    };
    testImg.onerror = function() {
        console.error('❌ Failed to load visualization image');
        vizLoading.style.display = 'none';
        vizError.style.display = 'block';
        document.getElementById('visualization-error-text').textContent = 
            'Failed to load visualization image. Please try regenerating.';
    };
    testImg.src = imgUrl;
}

// Open visualization in new tab
function openVisualizationInNewTab() {
    const vizImg = document.getElementById('visualization-img');
    if (vizImg && vizImg.src) {
        window.open(vizImg.src, '_blank');
    }
}

// Regenerate visualization
async function regenerateVisualization() {
    const vizLoading = document.getElementById('visualization-loading');
    const vizImage = document.getElementById('visualization-image');
    const vizError = document.getElementById('visualization-error');
    
    vizImage.style.display = 'none';
    vizError.style.display = 'none';
    vizLoading.style.display = 'block';
    
    try {
        console.log('[API] Requesting visualization regeneration...');
        const response = await fetch(`${API_BASE}/api/generate_visualization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success && data.filename) {
            console.log('[API] ✅ Visualization regenerated:', data.filename);
            displayVisualization(data.filename);
        } else {
            throw new Error(data.error || 'Failed to generate visualization');
        }
    } catch (error) {
        console.error('[API] ❌ Visualization regeneration failed:', error);
        vizLoading.style.display = 'none';
        vizError.style.display = 'block';
        document.getElementById('visualization-error-text').textContent = 
            'Failed to regenerate visualization: ' + error.message;
    }
}
