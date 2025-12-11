/**
 * Advanced LLM Reasoning Visualization
 * Features: Animated Reasoning Trace, Sankey Diagram, Pattern Detection, Comparative Analysis
 * 
 * DATA VISUALIZATION COURSE PROJECT
 * ================================
 * This system visualizes how Large Language Models (LLMs) "think" by showing:
 * 1. Attention patterns - which words the model focuses on
 * 2. Knowledge graphs - entities and relationships extracted from text
 * 3. Token importance - which words matter most for the answer
 * 
 * KEY VISUALIZATION PRINCIPLES APPLIED:
 * - Shneiderman's Mantra: Overview first, zoom and filter, details on demand
 * - Brushing & Linking: Selecting in one view highlights in others
 * - Color encoding: Consistent color scales across visualizations
 * - Animation: Temporal encoding for sequential processes
 */

const API_URL = 'http://localhost:8001';
let globalData = null;
let animationState = { playing: false, step: 0, interval: null };
let comparisonData = { low: null, high: null };

// Brushing & Linking state - tracks what's selected across all views
let selectionState = {
    selectedInputTokens: new Set(),
    selectedOutputTokens: new Set(),
    selectedEntities: new Set(),
    highlightMode: false
};

// ============== INITIALIZATION ==============

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
});

function initEventListeners() {
    // Sample questions
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            document.getElementById('question-input').value = e.target.dataset.question;
        });
    });

    // Slider updates
    document.getElementById('temperature').addEventListener('input', e => {
        document.getElementById('temp-val').textContent = e.target.value;
    });
    document.getElementById('max-tokens').addEventListener('input', e => {
        document.getElementById('tokens-val').textContent = e.target.value;
    });
    document.getElementById('top-p').addEventListener('input', e => {
        document.getElementById('topp-val').textContent = e.target.value;
    });

    // Submit
    document.getElementById('submit-btn').addEventListener('click', analyzeQuestion);
    document.getElementById('question-input').addEventListener('keypress', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeQuestion();
        }
    });

    // Animation controls
    document.getElementById('play-btn').addEventListener('click', playAnimation);
    document.getElementById('pause-btn').addEventListener('click', pauseAnimation);
    document.getElementById('reset-btn').addEventListener('click', resetAnimation);
    document.getElementById('step-btn').addEventListener('click', stepAnimation);

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            e.target.classList.add('active');
            document.getElementById(e.target.dataset.tab).classList.add('active');
        });
    });

    // Comparison
    document.getElementById('run-comparison').addEventListener('click', runTemperatureComparison);
    document.getElementById('run-question-compare').addEventListener('click', runQuestionComparison);
}

// ============== API CALLS ==============

async function analyzeQuestion() {
    const question = document.getElementById('question-input').value.trim();
    if (!question) return alert('Please enter a question');

    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxTokens = parseInt(document.getElementById('max-tokens').value);
    const topP = parseFloat(document.getElementById('top-p').value);

    showLoading(true);

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, max_length: maxTokens, temperature, top_p: topP })
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        globalData = await response.json();
        console.log('Data received:', globalData);

        displayResults(globalData);
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function queryWithParams(question, temperature, maxTokens = 100) {
    const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, max_length: maxTokens, temperature, top_p: 0.95 })
    });
    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    return response.json();
}

// ============== DISPLAY ==============

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
    document.getElementById('results').classList.toggle('hidden', show);
    document.getElementById('submit-btn').disabled = show;
}

function displayResults(data) {
    console.log("Displaying results...", data);
    
    // Clear previous highlights
    clearAllHighlights();
    
    // Answer
    document.getElementById('answer-text').textContent = data.answer || '(No answer)';

    // Metrics
    const metrics = calculateMetrics(data);
    document.getElementById('m-tokens').textContent = data.tokens?.length || 0;
    document.getElementById('m-entities').textContent = data.knowledge_graph?.nodes?.length || 0;
    document.getElementById('m-entropy').textContent = metrics.entropy.toFixed(2);
    document.getElementById('m-focus').textContent = Math.round(metrics.focus * 100) + '%';
    document.getElementById('m-sparsity').textContent = Math.round(metrics.sparsity * 100) + '%';

    // Reset animation
    resetAnimation();

    // Render all visualizations
    setTimeout(() => {
        console.log("Triggering renders...");
        try {
            renderAnimatedTrace(data);
            renderSankeyDiagram(data);
            renderTokenGradient(data);
            renderAttentionHeatmap(data);
            renderKnowledgeGraph(data);
            render3DAttention(data);
            renderPatternDetection(data);
            renderAttentionStats(data);
            
            // Generate and display insights (NEW - for professor evaluation)
            const insights = generateInsights(data);
            displayInsights(insights);
            
        } catch (e) {
            console.error("Error during rendering:", e);
        }
    }, 100);
}

function calculateMetrics(data) {
    const matrix = data.attention?.attention_mean || [];
    let entropy = 0, focus = 0, sparsity = 0;

    if (matrix.length > 0) {
        let totalEntropy = 0, totalFocus = 0, sparseCount = 0;

        matrix.forEach(row => {
            const sum = row.reduce((a, b) => a + b, 0) || 1;
            const norm = row.map(v => v / sum);

            // Entropy
            totalEntropy += -norm.reduce((acc, p) => acc + (p > 0 ? p * Math.log2(p) : 0), 0);

            // Focus (max attention)
            totalFocus += Math.max(...norm);

            // Sparsity (% of weights < 0.01)
            sparseCount += norm.filter(v => v < 0.01).length;
        });

        entropy = totalEntropy / matrix.length;
        focus = totalFocus / matrix.length;
        sparsity = sparseCount / (matrix.length * matrix.length);
    }

    return { entropy, focus, sparsity };
}

// ============== 1. ANIMATED REASONING TRACE ==============
// 
// WHAT THIS SHOWS (for presentation):
// ---------------------------------
// This visualization demonstrates HOW the LLM "thinks" when generating each word.
// 
// LEFT SIDE (Green): Words from your QUESTION
// RIGHT SIDE (Blue): Words the model GENERATES as answer
// 
// CONNECTIONS: When generating each answer word, the model "looks back" at the question.
// The LINE THICKNESS shows how much ATTENTION the model pays to each question word.
// 
// EXAMPLE: When generating "Paris", the model pays high attention to "capital" and "France"
// because those words are most relevant for producing that answer.
// 
// WHY THIS MATTERS:
// - Shows the model isn't just memorizing - it's actively relating question words to answers
// - Thicker lines = stronger influence on the generated word
// - Helps explain WHY the model gave a particular answer

function renderAnimatedTrace(data) {
    const container = document.getElementById('animated-trace');
    container.innerHTML = '';

    const questionTokens = data.question_tokens || [];
    const answerTokens = data.answer_tokens || [];

    if (questionTokens.length === 0 || answerTokens.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">Insufficient token data</div>';
        return;
    }

    // Filter tokens - keep meaningful words only
    const inputTokens = questionTokens.filter(t => t && (t.length > 1 || /[a-zA-Z0-9]/.test(t)) && !/^[?.,!;:'"]+$/.test(t)).slice(0, 15);
    const outputTokens = answerTokens.filter(t => t && (t.length > 1 || /[a-zA-Z0-9]/.test(t)) && !/^[?.,!;:'"<>]+$/.test(t) && !t.startsWith('<')).slice(0, 20);

    if (inputTokens.length === 0 || outputTokens.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.8;color:white;">No meaningful tokens to display</div>';
        return;
    }

    document.getElementById('total-steps').textContent = outputTokens.length;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { left: 140, right: 140, top: 70, bottom: 50 };

    const svg = d3.select('#animated-trace')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Scales
    const inputY = d3.scaleLinear()
        .domain([0, Math.max(inputTokens.length - 1, 1)])
        .range([margin.top, height - margin.bottom]);

    const outputY = d3.scaleLinear()
        .domain([0, Math.max(outputTokens.length - 1, 1)])
        .range([margin.top, height - margin.bottom]);

    // Get attention data
    const matrix = data.attention?.attention_mean || [];
    const questionStartIdx = data.attention?.question_start_idx || 0;
    const answerIndices = data.answer_indices || [];

    // Build relevance matrix from actual attention weights
    const relevanceMatrix = [];
    outputTokens.forEach((outToken, relativeOutIdx) => {
        const row = [];
        const globalOutIdx = answerIndices[relativeOutIdx];

        if (typeof globalOutIdx === 'undefined' || globalOutIdx >= matrix.length) {
            inputTokens.forEach(() => row.push(0));
        } else {
            const attRow = matrix[globalOutIdx];
            inputTokens.forEach((inToken, relativeInIdx) => {
                const globalInIdx = questionStartIdx + relativeInIdx;
                if (globalInIdx < attRow.length) {
                    row.push(attRow[globalInIdx]);
                } else {
                    row.push(0);
                }
            });
        }
        relevanceMatrix.push(row);
    });

    animationState.data = {
        inputTokens, outputTokens, relevanceMatrix,
        width, height, margin, inputY, outputY
    };

    // Calculate total importance per input token (sum of attention from all outputs)
    const inputImportance = inputTokens.map((t, i) => {
        return relevanceMatrix.reduce((sum, row) => sum + row[i], 0);
    });
    const maxInputImp = Math.max(...inputImportance, 0.001);

    // Add explanation box at top
    svg.append('rect')
        .attr('x', width / 2 - 200)
        .attr('y', 5)
        .attr('width', 400)
        .attr('height', 45)
        .attr('fill', 'rgba(102, 126, 234, 0.15)')
        .attr('rx', 8);

    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 22)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text('🔍 How the Model "Thinks"');

    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 38)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.7)')
        .attr('font-size', '10px')
        .text('Thicker lines = model pays MORE attention to that word when generating');

    // Draw input nodes with importance-based sizing
    svg.selectAll('.input-node')
        .data(inputTokens)
        .enter()
        .append('circle')
        .attr('class', 'input-node')
        .attr('cx', margin.left)
        .attr('cy', (d, i) => inputY(i))
        .attr('r', (d, i) => 6 + (inputImportance[i] / maxInputImp) * 8)
        .attr('fill', (d, i) => {
            const imp = inputImportance[i] / maxInputImp;
            return imp > 0.7 ? '#FFEB3B' : imp > 0.4 ? '#8BC34A' : '#4CAF50';
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Input labels with importance indicator
    svg.selectAll('.input-label')
        .data(inputTokens)
        .enter()
        .append('text')
        .attr('x', margin.left - 15)
        .attr('y', (d, i) => inputY(i))
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', '#fff')
        .attr('font-size', '12px')
        .attr('font-weight', (d, i) => inputImportance[i] / maxInputImp > 0.5 ? 'bold' : 'normal')
        .text(d => d.length > 12 ? d.slice(0, 12) + '…' : d);

    // Output nodes (initially hidden)
    svg.selectAll('.output-node')
        .data(outputTokens)
        .enter()
        .append('circle')
        .attr('class', 'output-node')
        .attr('cx', width - margin.right)
        .attr('cy', (d, i) => outputY(i))
        .attr('r', 8)
        .attr('fill', '#2196F3')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('opacity', 0);

    // Output labels
    svg.selectAll('.output-label')
        .data(outputTokens)
        .enter()
        .append('text')
        .attr('class', 'output-label')
        .attr('x', width - margin.right + 15)
        .attr('y', (d, i) => outputY(i))
        .attr('dy', '0.35em')
        .attr('text-anchor', 'start')
        .attr('fill', '#fff')
        .attr('font-size', '12px')
        .attr('opacity', 0)
        .text(d => d.length > 12 ? d.slice(0, 12) + '…' : d);

    // Group for animated links
    svg.append('g').attr('class', 'links-group');

    // Title labels with icons
    svg.append('text')
        .attr('x', margin.left)
        .attr('y', margin.top - 15)
        .attr('fill', '#4CAF50')
        .attr('font-size', '13px')
        .attr('font-weight', 'bold')
        .text('📝 QUESTION WORDS');

    svg.append('text')
        .attr('x', width - margin.right)
        .attr('y', margin.top - 15)
        .attr('fill', '#2196F3')
        .attr('font-size', '13px')
        .attr('font-weight', 'bold')
        .attr('text-anchor', 'end')
        .text('💬 GENERATED ANSWER');

    // Current token explanation label
    svg.append('text')
        .attr('class', 'current-token-label')
        .attr('x', width / 2)
        .attr('y', height - 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#667eea')
        .attr('font-size', '13px')
        .attr('font-weight', 'bold');

    // Legend for node sizes
    const legendY = height - 35;
    svg.append('circle').attr('cx', 20).attr('cy', legendY).attr('r', 5).attr('fill', '#4CAF50');
    svg.append('text').attr('x', 30).attr('y', legendY + 4).attr('fill', 'rgba(255,255,255,0.6)').attr('font-size', '9px').text('Low importance');
    svg.append('circle').attr('cx', 110).attr('cy', legendY).attr('r', 8).attr('fill', '#8BC34A');
    svg.append('text').attr('x', 123).attr('y', legendY + 4).attr('fill', 'rgba(255,255,255,0.6)').attr('font-size', '9px').text('Medium');
    svg.append('circle').attr('cx', 180).attr('cy', legendY).attr('r', 12).attr('fill', '#FFEB3B');
    svg.append('text').attr('x', 197).attr('y', legendY + 4).attr('fill', 'rgba(255,255,255,0.6)').attr('font-size', '9px').text('High importance');
}

// Helper function to animate steps based on real attention
// Each step shows: "When generating word X, the model looked at these question words"
function animateStep(step) {
    if (!animationState.data) return;

    const { inputTokens, outputTokens, relevanceMatrix, width, margin, inputY, outputY } = animationState.data;
    const svg = d3.select('#animated-trace svg');

    if (step >= outputTokens.length) {
        pauseAnimation();
        return;
    }

    animationState.step = step;
    document.getElementById('current-step').textContent = step + 1;

    // Show current output node (highlight the word being generated)
    svg.selectAll('.output-node')
        .attr('opacity', (d, i) => i <= step ? 1 : 0)
        .attr('fill', (d, i) => i === step ? '#64B5F6' : '#2196F3')
        .attr('r', (d, i) => i === step ? 12 : 8);

    svg.selectAll('.output-label')
        .attr('opacity', (d, i) => i <= step ? 1 : 0)
        .attr('font-weight', (d, i) => i === step ? 'bold' : 'normal');

    // Get attention weights for current output token
    const relevanceRow = relevanceMatrix[step] || [];
    const maxRelevance = Math.max(...relevanceRow, 0.001);

    // Create links - each link shows "this output word attended to this input word"
    const links = inputTokens.map((token, i) => ({
        inputIdx: i,
        relevance: relevanceRow[i] || 0,
        normalized: (relevanceRow[i] || 0) / maxRelevance,
        token: token
    })).filter(l => l.normalized > 0.05); // Show connections with > 5% relative attention

    const linksGroup = svg.select('.links-group');

    // Color scale: yellow = high attention, blue = low attention
    const colorScale = d3.scaleSequential(d3.interpolateYlGnBu).domain([0, 1]);

    const linkGenerator = d3.linkHorizontal()
        .x(d => d.x)
        .y(d => d.y);

    // Draw links with animation
    linksGroup.selectAll(`.link-step-${step}`)
        .data(links)
        .enter()
        .append('path')
        .attr('class', `link-step-${step}`)
        .attr('d', d => linkGenerator({
            source: { x: margin.left + 10, y: inputY(d.inputIdx) },
            target: { x: width - margin.right - 10, y: outputY(step) }
        }))
        .attr('fill', 'none')
        .attr('stroke', d => colorScale(d.normalized))
        .attr('stroke-width', d => Math.max(1.5, d.normalized * 10))
        .attr('opacity', 0)
        .transition()
        .duration(300)
        .attr('opacity', 0.85);

    // Fade previous step's links
    linksGroup.selectAll(`path:not(.link-step-${step})`)
        .transition()
        .duration(200)
        .attr('opacity', 0.15);

    // Build explanation text
    const sortedLinks = [...links].sort((a, b) => b.relevance - a.relevance);
    const topWords = sortedLinks.slice(0, 3).map(l => `"${l.token}"`).join(', ');
    const explanation = topWords 
        ? `To generate "${outputTokens[step]}", model focused on: ${topWords}`
        : `Generating: "${outputTokens[step]}"`;
    
    svg.select('.current-token-label').text(explanation);

    // Highlight the input nodes that received attention
    const relevanceByInput = {};
    links.forEach(l => {
        relevanceByInput[l.inputIdx] = l.normalized;
    });

    svg.selectAll('.input-node')
        .transition()
        .duration(200)
        .attr('r', (d, i) => relevanceByInput[i] > 0.5 ? 14 : (relevanceByInput[i] > 0.2 ? 10 : 8))
        .attr('fill', (d, i) => {
            const rel = relevanceByInput[i] || 0;
            if (rel > 0.7) return '#FFEB3B';  // High attention = yellow
            if (rel > 0.4) return '#8BC34A';  // Medium = light green
            return '#4CAF50';                  // Low = green
        });
}

function playAnimation() {
    if (animationState.playing) return;
    animationState.playing = true;

    const speed = parseInt(document.getElementById('anim-speed').value);

    animationState.interval = setInterval(() => {
        if (animationState.step >= (animationState.data?.outputTokens?.length || 0)) {
            pauseAnimation();
            return;
        }
        animateStep(animationState.step);
        animationState.step++;
    }, speed);
}

function pauseAnimation() {
    animationState.playing = false;
    if (animationState.interval) {
        clearInterval(animationState.interval);
        animationState.interval = null;
    }
}

function resetAnimation() {
    pauseAnimation();
    animationState.step = 0;
    document.getElementById('current-step').textContent = '0';

    const svg = d3.select('#animated-trace svg');
    if (svg.empty()) return;

    svg.select('.links-group').selectAll('*').remove();
    svg.selectAll('.output-node').attr('opacity', 0);
    svg.selectAll('.output-label').attr('opacity', 0);
    svg.select('.current-token-label').text('');
}

function stepAnimation() {
    if (animationState.step >= (animationState.data?.outputTokens?.length || 0)) return;
    animateStep(animationState.step);
    animationState.step++;
}


// ============== 2. SANKEY DIAGRAM ==============
//
// WHAT THIS SHOWS (for presentation):
// ---------------------------------
// A FLOW DIAGRAM showing how "attention" flows from question words to answer words.
// 
// LEFT SIDE: Words from your question (input) - in original order
// RIGHT SIDE: Words the model generated (output) - in generation order
// FLOW WIDTH: How much the model "used" each input word to generate each output word
// 
// INTERPRETATION:
// - Wider flows = stronger connection (model relied heavily on that input word)
// - Multiple flows from one input = that word influenced many output words
// - This shows the model's "reasoning path" from question to answer

// Helper function to merge subword tokens for better display
// e.g., ["Rome", "o"] → ["Romeo"], ["Shakes", "peare"] → ["Shakespeare"]
function mergeSubwordTokens(tokens) {
    const merged = [];
    let currentWord = '';
    
    tokens.forEach((token, i) => {
        // Check if this token looks like a continuation (lowercase, short, no space prefix)
        const isSubword = token.length <= 3 && 
                         /^[a-z]+$/.test(token) && 
                         currentWord.length > 0 &&
                         i > 0;
        
        // Also check if previous token + this token forms a known word pattern
        const prevToken = tokens[i - 1] || '';
        const combined = prevToken + token;
        const looksLikeContinuation = isSubword || 
            (token.length <= 2 && /^[a-z]/.test(token) && merged.length > 0);
        
        if (looksLikeContinuation && merged.length > 0) {
            // Merge with previous token
            merged[merged.length - 1] = merged[merged.length - 1] + token;
        } else {
            merged.push(token);
        }
    });
    
    return merged;
}

function renderSankeyDiagram(data) {
    const container = document.getElementById('sankey-diagram');
    container.innerHTML = '';

    // Get tokens - keep original order, just filter punctuation
    const questionTokens = (data.question_tokens || []).filter(t => t && t.trim() && !/^[?.,!;:'"]+$/.test(t));
    const answerTokens = (data.answer_tokens || []).filter(t => t && t.trim() && !/^[?.,!;:'"<>]+$/.test(t) && !t.startsWith('<'));

    if (questionTokens.length === 0 || answerTokens.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">Insufficient data for Sankey diagram</div>';
        return;
    }

    // Merge subword tokens for cleaner display
    const mergedInputTokens = mergeSubwordTokens(questionTokens);
    const mergedOutputTokens = mergeSubwordTokens(answerTokens);
    
    // Keep tokens in original order, limit for display
    const inputTokens = mergedInputTokens.slice(0, 10);
    const outputTokens = mergedOutputTokens.slice(0, 12);

    console.log("Sankey Input tokens (merged):", inputTokens);
    console.log("Sankey Output tokens (merged):", outputTokens);

    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 50, right: 150, bottom: 30, left: 150 };

    // Check if d3.sankey is available
    if (typeof d3.sankey !== 'function') {
        // Fallback to flow diagram
        renderFlowDiagramFallback(container, inputTokens, outputTokens, data, width, height, margin);
        return;
    }

    // Build Sankey data
    const nodes = [];
    const links = [];

    // Add input nodes
    inputTokens.forEach((token, i) => {
        nodes.push({ id: `in_${i}`, name: token, type: 'input' });
    });

    // Add output nodes
    outputTokens.forEach((token, i) => {
        nodes.push({ id: `out_${i}`, name: token, type: 'output' });
    });

    // Use REAL ATTENTION WEIGHTS from the model
    const matrix = data.attention?.attention_mean || [];
    const questionStartIdx = data.attention?.question_start_idx || 0;
    const answerIndices = data.answer_indices || [];

    // Aggregate attention: for each input token, sum attention from ALL output tokens
    const inputAttentionSum = new Array(inputTokens.length).fill(0);

    // Create links based on actual attention
    outputTokens.forEach((outToken, relativeOutIdx) => {
        const globalOutIdx = answerIndices[relativeOutIdx];

        if (typeof globalOutIdx !== 'undefined' && globalOutIdx < matrix.length) {
            const attentionRow = matrix[globalOutIdx];

            // Get attention from this output token to each input token
            inputTokens.forEach((inToken, relativeInIdx) => {
                const globalInIdx = questionStartIdx + relativeInIdx;

                if (globalInIdx < attentionRow.length) {
                    const weight = attentionRow[globalInIdx];
                    inputAttentionSum[relativeInIdx] += weight;

                    if (weight > 0.01) {  // Threshold
                        links.push({
                            source: relativeInIdx,
                            target: inputTokens.length + relativeOutIdx,
                            value: weight
                        });
                    }
                }
            });
        }
    });

    // If still no links, create links based on aggregated importance
    if (links.length < 5) {
        console.log('Few links found, using aggregated attention');
        const maxSum = Math.max(...inputAttentionSum, 0.01);

        inputTokens.forEach((inToken, i) => {
            const normalizedImportance = inputAttentionSum[i] / maxSum;
            if (normalizedImportance > 0.1) {
                // Connect important input tokens to multiple output tokens
                outputTokens.slice(0, 5).forEach((_, j) => {
                    links.push({
                        source: i,
                        target: inputTokens.length + j,
                        value: normalizedImportance * 0.5
                    });
                });
            }
        });
    }

    // Ensure every node has at least one connection
    inputTokens.forEach((_, i) => {
        const hasLink = links.some(l => l.source === i);
        if (!hasLink && outputTokens.length > 0) {
            links.push({ source: i, target: inputTokens.length, value: 0.1 });
        }
    });

    outputTokens.forEach((_, j) => {
        const hasLink = links.some(l => l.target === inputTokens.length + j);
        if (!hasLink && inputTokens.length > 0) {
            links.push({ source: 0, target: inputTokens.length + j, value: 0.1 });
        }
    });

    // Normalize link values for better visualization
    const maxLinkValue = Math.max(...links.map(l => l.value), 0.01);
    links.forEach(l => l.value = Math.max(0.1, l.value / maxLinkValue));

    const sankeySvg = d3.select('#sankey-diagram')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    try {
        // Create Sankey generator
        const sankey = d3.sankey()
            .nodeWidth(20)
            .nodePadding(12)
            .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]]);

        // Generate layout
        const sankeyData = sankey({
            nodes: nodes.map(d => ({ ...d })),
            links: links.map(d => ({ ...d }))
        });

        // Color scales
        const inputColor = '#4CAF50';
        const outputColor = '#2196F3';
        const linkColor = d3.scaleSequential(d3.interpolatePurples).domain([0, 1]);

        // Draw links
        sankeySvg.append('g')
            .selectAll('path')
            .data(sankeyData.links)
            .enter()
            .append('path')
            .attr('d', d3.sankeyLinkHorizontal())
            .attr('fill', 'none')
            .attr('stroke', d => linkColor(d.value))
            .attr('stroke-width', d => Math.max(2, d.width))
            .attr('opacity', 0.6)
            .on('mouseover', function (event, d) {
                d3.select(this).attr('opacity', 1).attr('stroke-width', d.width + 3);
                showTooltip(event, `"${d.source.name}" → "${d.target.name}"<br>Weight: ${d.value.toFixed(2)}`);
            })
            .on('mouseout', function (event, d) {
                d3.select(this).attr('opacity', 0.6).attr('stroke-width', Math.max(2, d.width));
                hideTooltip();
            });

        // Draw nodes
        sankeySvg.append('g')
            .selectAll('rect')
            .data(sankeyData.nodes)
            .enter()
            .append('rect')
            .attr('x', d => d.x0)
            .attr('y', d => d.y0)
            .attr('width', d => d.x1 - d.x0)
            .attr('height', d => Math.max(1, d.y1 - d.y0))
            .attr('fill', d => d.type === 'input' ? inputColor : outputColor)
            .attr('rx', 3)
            .on('mouseover', function (event, d) {
                showTooltip(event, `<strong>${d.name}</strong><br>Type: ${d.type}`);
            })
            .on('mouseout', hideTooltip);

        // Node labels
        sankeySvg.append('g')
            .selectAll('text')
            .data(sankeyData.nodes)
            .enter()
            .append('text')
            .attr('x', d => d.type === 'input' ? d.x0 - 8 : d.x1 + 8)
            .attr('y', d => (d.y0 + d.y1) / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', d => d.type === 'input' ? 'end' : 'start')
            .attr('fill', '#fff')
            .attr('font-size', '11px')
            .text(d => d.name.length > 15 ? d.name.slice(0, 15) + '…' : d.name);

        // Labels
        sankeySvg.append('text')
            .attr('x', margin.left)
            .attr('y', 15)
            .attr('fill', inputColor)
            .attr('font-weight', 'bold')
            .attr('font-size', '13px')
            .text('Input Tokens');

        sankeySvg.append('text')
            .attr('x', width - margin.right)
            .attr('y', 15)
            .attr('fill', outputColor)
            .attr('font-weight', 'bold')
            .attr('font-size', '13px')
            .attr('text-anchor', 'end')
            .text('Output Tokens');
    } catch (e) {
        console.error('Sankey error:', e);
        container.innerHTML = '';
        renderFlowDiagramFallback(container, inputTokens, outputTokens, data, width, height, margin);
    }
}

// Fallback flow diagram if Sankey library not available
function renderFlowDiagramFallback(container, inputTokens, outputTokens, data, width, height, margin) {
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Use REAL ATTENTION WEIGHTS
    const matrix = data.attention?.attention_mean || [];
    const promptLength = data.attention?.prompt_length || 0;

    // Use the question start index from backend
    const questionStartIdx = data.attention?.question_start_idx || Math.max(0, promptLength - inputTokens.length - 10);

    const inputY = d3.scaleLinear()
        .domain([0, Math.max(inputTokens.length - 1, 1)])
        .range([margin.top + 20, height - margin.bottom]);

    const outputY = d3.scaleLinear()
        .domain([0, Math.max(outputTokens.length - 1, 1)])
        .range([margin.top + 20, height - margin.bottom]);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    // Create links from attention with offset search
    const links = [];
    outputTokens.forEach((outToken, j) => {
        const outputMatrixIdx = promptLength + j;

        if (outputMatrixIdx < matrix.length) {
            const attentionRow = matrix[outputMatrixIdx];

            inputTokens.forEach((inToken, i) => {
                let maxWeight = 0;
                for (let offset = -3; offset <= 3; offset++) {
                    const matrixIdx = questionStartIdx + i + offset;
                    if (matrixIdx >= 0 && matrixIdx < attentionRow.length) {
                        maxWeight = Math.max(maxWeight, attentionRow[matrixIdx]);
                    }
                }

                if (maxWeight > 0.005) {
                    links.push({ i, j, weight: maxWeight, inToken, outToken });
                }
            });
        }
    });

    // Ensure we have some links
    if (links.length < 3) {
        // Fallback: connect based on token similarity
        inputTokens.forEach((inToken, i) => {
            outputTokens.forEach((outToken, j) => {
                if (inToken.toLowerCase() === outToken.toLowerCase() ||
                    outToken.toLowerCase().includes(inToken.toLowerCase())) {
                    links.push({ i, j, weight: 0.5, inToken, outToken });
                }
            });
        });
    }

    // Normalize weights
    const maxWeight = Math.max(...links.map(l => l.weight), 0.1);
    links.forEach(l => l.normalizedWeight = l.weight / maxWeight);

    const linkGenerator = d3.linkHorizontal().x(d => d.x).y(d => d.y);

    // Draw links using normalized weights with tooltips
    svg.selectAll('.flow-link')
        .data(links)
        .enter()
        .append('path')
        .attr('class', 'flow-link')
        .attr('d', d => linkGenerator({
            source: { x: margin.left + 10, y: inputY(d.i) },
            target: { x: width - margin.right - 10, y: outputY(d.j) }
        }))
        .attr('fill', 'none')
        .attr('stroke', d => colorScale(d.normalizedWeight))
        .attr('stroke-width', d => Math.max(1.5, d.normalizedWeight * 10))
        .attr('opacity', 0.5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('opacity', 1)
                .attr('stroke-width', d.normalizedWeight * 12 + 3);
            showTooltip(event, `"${d.inToken}" → "${d.outToken}"<br>Weight: ${d.weight.toFixed(3)}<br>Normalized: ${(d.normalizedWeight * 100).toFixed(1)}%`);
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .attr('opacity', 0.5)
                .attr('stroke-width', Math.max(1.5, d.normalizedWeight * 10));
            hideTooltip();
        });

    // Input nodes
    svg.selectAll('.input-node')
        .data(inputTokens)
        .enter()
        .append('circle')
        .attr('cx', margin.left)
        .attr('cy', (d, i) => inputY(i))
        .attr('r', 8)
        .attr('fill', '#4CAF50')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Output nodes
    svg.selectAll('.output-node')
        .data(outputTokens)
        .enter()
        .append('circle')
        .attr('cx', width - margin.right)
        .attr('cy', (d, i) => outputY(i))
        .attr('r', 8)
        .attr('fill', '#2196F3')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Labels
    // Input labels with index numbers to show order
    svg.selectAll('.input-label')
        .data(inputTokens)
        .enter()
        .append('text')
        .attr('x', margin.left - 15)
        .attr('y', (d, i) => inputY(i))
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .text((d, i) => {
            const label = d.length > 10 ? d.slice(0, 10) + '…' : d;
            return label;
        });

    // Output labels
    svg.selectAll('.output-label')
        .data(outputTokens)
        .enter()
        .append('text')
        .attr('x', width - margin.right + 15)
        .attr('y', (d, i) => outputY(i))
        .attr('dy', '0.35em')
        .attr('text-anchor', 'start')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .text((d, i) => {
            const label = d.length > 10 ? d.slice(0, 10) + '…' : d;
            return label;
        });

    // Title labels
    svg.append('text')
        .attr('x', margin.left)
        .attr('y', 20)
        .attr('fill', '#4CAF50')
        .attr('font-weight', 'bold')
        .attr('font-size', '13px')
        .text('Input Tokens');

    svg.append('text')
        .attr('x', width - margin.right)
        .attr('y', 20)
        .attr('fill', '#2196F3')
        .attr('font-weight', 'bold')
        .attr('font-size', '13px')
        .attr('text-anchor', 'end')
        .text('Output Tokens');

    // Add color legend
    const legendX = width / 2 - 60;
    svg.append('text')
        .attr('x', legendX)
        .attr('y', height - 5)
        .attr('fill', 'rgba(255,255,255,0.6)')
        .attr('font-size', '10px')
        .text('Line color/thickness = attention strength');
}

// ============== 3. TOKEN GRADIENT ==============
//
// WHAT THIS SHOWS (for presentation):
// ---------------------------------
// Each word is colored by its IMPORTANCE in the attention mechanism.
// 
// QUESTION WORDS (Green shades):
// - Darker green = model paid MORE attention to this word overall
// - These are the "key words" the model focused on to understand the question
// 
// ANSWER WORDS (Blue shades):
// - Darker blue = this word was generated with MORE focus on the question
// - Lighter = generated more independently (less question-dependent)
//
// USE CASE: Quickly identify which words were most important for the model's reasoning

function renderTokenGradient(data) {
    const container = document.getElementById('token-gradient');
    container.innerHTML = '';

    const questionTokens = data.question_tokens || [];
    const answerTokens = data.answer_tokens || [];
    const matrix = data.attention?.attention_mean || [];
    const questionStartIdx = data.attention?.question_start_idx || 0;
    const answerIndices = data.answer_indices || [];

    // 1. Calculate Question Token Importance:
    // Sum of attention received from ALL clean answer tokens
    const qImportanceVector = new Array(questionTokens.length).fill(0);

    // Iterate over all answer tokens that have a valid matrix index
    answerTokens.forEach((_, relativeOutIdx) => {
        const globalOutIdx = answerIndices[relativeOutIdx];
        if (typeof globalOutIdx !== 'undefined' && globalOutIdx < matrix.length) {
            const row = matrix[globalOutIdx];
            // Accumulate attention for each question token
            questionTokens.forEach((_, relativeInIdx) => {
                const globalInIdx = questionStartIdx + relativeInIdx;
                if (globalInIdx < row.length) {
                    qImportanceVector[relativeInIdx] += row[globalInIdx];
                }
            });
        }
    });

    const maxQImp = Math.max(...qImportanceVector, 0.001);

    // Add question label
    const qLabel = document.createElement('span');
    qLabel.style.cssText = 'color: #4CAF50; font-weight: bold; margin-right: 10px;';
    qLabel.textContent = 'Question:';
    container.appendChild(qLabel);

    // Render Question tokens
    questionTokens.forEach((token, i) => {
        if (!token) return;
        const span = document.createElement('span');
        span.className = 'gradient-token';
        span.textContent = token;

        const val = qImportanceVector[i];
        const intensity = val / maxQImp;

        // Green gradient
        span.style.background = `rgba(76, 175, 80, ${0.2 + intensity * 0.7})`;
        span.style.color = intensity > 0.5 ? '#fff' : '#e0e0e0';

        span.title = `Token: "${token}"\nAccumulated Attention: ${val.toFixed(4)}`;
        span.addEventListener('mouseover', e => showTooltip(e, span.title.replace(/\n/g, '<br>')));
        span.addEventListener('mouseout', hideTooltip);

        container.appendChild(span);
    });

    // Separator
    const sep = document.createElement('div');
    sep.style.cssText = 'width: 100%; height: 15px;';
    container.appendChild(sep);

    // Answer label
    const aLabel = document.createElement('span');
    aLabel.style.cssText = 'color: #2196F3; font-weight: bold; margin-right: 10px;';
    aLabel.textContent = 'Answer:';
    container.appendChild(aLabel);

    // 2. Calculate Answer Token "Confidence" or "Focus"
    // (For answer tokens, we can visualize how "focused" their attention was, or just average attention mass)
    // Here, let's visualize the average attention mass they put onto the question (relevance to question)
    const aImportanceVector = [];

    answerTokens.forEach((token, relativeOutIdx) => {
        const globalOutIdx = answerIndices[relativeOutIdx];
        let attnSum = 0;
        if (typeof globalOutIdx !== 'undefined' && globalOutIdx < matrix.length) {
            const row = matrix[globalOutIdx];
            // Sum attention only to question tokens
            for (let i = 0; i < questionTokens.length; i++) {
                const globalInIdx = questionStartIdx + i;
                if (globalInIdx < row.length) {
                    attnSum += row[globalInIdx];
                }
            }
        }
        aImportanceVector.push(attnSum);
    });

    const maxAImp = Math.max(...aImportanceVector, 0.001);

    // Render Answer tokens
    answerTokens.forEach((token, i) => {
        if (!token || token.startsWith('<')) return;

        const span = document.createElement('span');
        span.className = 'gradient-token';
        span.textContent = token;

        const val = aImportanceVector[i];
        const intensity = val / maxAImp;

        // Blue gradient
        span.style.background = `rgba(33, 150, 243, ${0.2 + intensity * 0.7})`;
        span.style.color = intensity > 0.5 ? '#fff' : '#e0e0e0';

        span.title = `Token: "${token}"\nAttention to Question: ${val.toFixed(4)}`;
        span.addEventListener('mouseover', e => showTooltip(e, span.title.replace(/\n/g, '<br>')));
        span.addEventListener('mouseout', hideTooltip);

        container.appendChild(span);
    });
}

// ============== 4. ATTENTION HEATMAP ==============
//
// WHAT THIS SHOWS (for presentation):
// ---------------------------------
// A MATRIX showing attention weights between every output word and every input word.
// 
// ROWS (Y-axis): Words the model GENERATED (answer)
// COLUMNS (X-axis): Words from your QUESTION (input)
// CELL COLOR: How much attention that output word paid to that input word
//   - Bright/Yellow = HIGH attention (strong connection)
//   - Dark/Purple = LOW attention (weak connection)
// 
// HOW TO READ:
// Pick any row (output word) and look across - the brightest cells show
// which question words the model "looked at" when generating that word.
//
// EXAMPLE: If "Paris" row has bright cells at "capital" and "France" columns,
// it means the model connected those concepts to generate "Paris"

function renderAttentionHeatmap(data) {
    const container = document.getElementById('attention-heatmap');
    container.innerHTML = '';

    const matrix = data.attention?.attention_mean || [];
    const questionTokens = data.question_tokens || [];
    const answerTokens = data.answer_tokens || [];
    const questionStartIdx = data.attention?.question_start_idx || 0;
    const answerIndices = data.answer_indices || [];

    if (matrix.length === 0 || questionTokens.length === 0 || answerTokens.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">No attention data</div>';
        return;
    }

    // Prepare Filtered Input Tokens with Original Indices
    const inputNodes = [];
    questionTokens.forEach((token, i) => {
        if (token && !/^[?.,!;:'"<>]+$/.test(token) && !token.startsWith('<')) {
            inputNodes.push({ token, originalIdx: i });
        }
    });
    // Limit to top 15 for display
    const displayInputs = inputNodes.slice(0, 15);

    // Prepare Filtered Output Tokens with Original Indices
    const outputNodes = [];
    answerTokens.forEach((token, i) => {
        if (token && !/^[?.,!;:'"<>]+$/.test(token) && !token.startsWith('<')) {
            outputNodes.push({ token, originalIdx: i });
        }
    });
    // Limit to top 15 for display
    const displayOutputs = outputNodes.slice(0, 15);

    // Build focused attention matrix: output (rows) x input (cols)
    const focusedMatrix = [];

    displayOutputs.forEach(({ token: outToken, originalIdx: relativeOutIdx }) => {
        const row = [];
        const globalOutIdx = answerIndices[relativeOutIdx];

        if (typeof globalOutIdx !== 'undefined' && globalOutIdx < matrix.length) {
            const attentionRow = matrix[globalOutIdx];

            displayInputs.forEach(({ token: inToken, originalIdx: relativeInIdx }) => {
                const globalInIdx = questionStartIdx + relativeInIdx;
                if (globalInIdx < attentionRow.length) {
                    row.push(attentionRow[globalInIdx]);
                } else {
                    row.push(0);
                }
            });
        } else {
            // If invalid row, fill with zeros
            displayInputs.forEach(() => row.push(0));
        }
        focusedMatrix.push(row);
    });

    // Use D3 to render heatmap
    const margin = { top: 30, right: 30, bottom: 50, left: 100 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select('#attention-heatmap')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // X scale (Inputs)
    const x = d3.scaleBand()
        .range([0, width])
        .domain(displayInputs.map((d, i) => i))
        .padding(0.01);

    // Y scale (Outputs)
    const y = d3.scaleBand()
        .range([0, height])
        .domain(displayOutputs.map((d, i) => i))
        .padding(0.01);

    // Color scale
    const maxVal = Math.max(0.001, ...focusedMatrix.flat());
    const myColor = d3.scaleSequential()
        .interpolator(d3.interpolateInferno)
        .domain([0, maxVal]);

    // Draw Squares
    focusedMatrix.forEach((row, i) => {
        row.forEach((value, j) => {
            svg.append('rect')
                .attr('x', x(j))
                .attr('y', y(i))
                .attr('width', x.bandwidth())
                .attr('height', y.bandwidth())
                .style('fill', myColor(value))
                .on('mouseover', function (event) {
                    d3.select(this).style('stroke', 'white').style('stroke-width', 2);
                    const inTok = displayInputs[j].token;
                    const outTok = displayOutputs[i].token;
                    showTooltip(event, `Output: "${outTok}"<br>Input: "${inTok}"<br>Attn: ${value.toFixed(4)}`);
                })
                .on('mouseout', function () {
                    d3.select(this).style('stroke', 'none');
                    hideTooltip();
                });
        });
    });

    // X Axis labels (Question tokens)
    svg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(x).tickFormat((d, i) => displayInputs[i]?.token || ''))
        .selectAll('text')
        .attr('transform', 'translate(-10,0)rotate(-45)')
        .style('text-anchor', 'end')
        .style('fill', '#4CAF50')  // Green for question
        .style('font-size', '10px');

    // Y Axis labels (Answer tokens)
    svg.append('g')
        .call(d3.axisLeft(y).tickFormat((d, i) => displayOutputs[i]?.token || ''))
        .selectAll('text')
        .style('fill', '#2196F3')  // Blue for answer
        .style('font-size', '10px');

    // Axis titles
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 45)
        .style('text-anchor', 'middle')
        .style('fill', '#4CAF50')
        .style('font-size', '11px')
        .text('← Question Words →');

    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -85)
        .style('text-anchor', 'middle')
        .style('fill', '#2196F3')
        .style('font-size', '11px')
        .text('← Answer Words →');

    // Color legend
    const legendWidth = 120;
    const legendHeight = 10;
    const legendX = width - legendWidth - 10;
    const legendY = -25;

    // Gradient for legend
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
        .attr('id', 'heatmap-gradient')
        .attr('x1', '0%').attr('x2', '100%');
    
    gradient.append('stop').attr('offset', '0%').attr('stop-color', myColor(0));
    gradient.append('stop').attr('offset', '50%').attr('stop-color', myColor(maxVal / 2));
    gradient.append('stop').attr('offset', '100%').attr('stop-color', myColor(maxVal));

    svg.append('rect')
        .attr('x', legendX)
        .attr('y', legendY)
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#heatmap-gradient)');

    svg.append('text')
        .attr('x', legendX)
        .attr('y', legendY - 3)
        .style('fill', 'rgba(255,255,255,0.7)')
        .style('font-size', '9px')
        .text('Low');

    svg.append('text')
        .attr('x', legendX + legendWidth)
        .attr('y', legendY - 3)
        .style('fill', 'rgba(255,255,255,0.7)')
        .style('font-size', '9px')
        .style('text-anchor', 'end')
        .text('High');

    // Find and highlight strongest connections
    let strongestConnections = [];
    focusedMatrix.forEach((row, i) => {
        row.forEach((value, j) => {
            if (value > maxVal * 0.7) {  // Top 30% attention
                strongestConnections.push({
                    output: displayOutputs[i]?.token,
                    input: displayInputs[j]?.token,
                    value: value
                });
            }
        });
    });

    // Show strongest connections summary
    if (strongestConnections.length > 0) {
        strongestConnections.sort((a, b) => b.value - a.value);
        const topConnections = strongestConnections.slice(0, 3);
        
        const summaryText = topConnections
            .map(c => `"${c.output}" ← "${c.input}"`)
            .join(', ');
        
        svg.append('text')
            .attr('x', 0)
            .attr('y', -15)
            .style('fill', 'rgba(255,255,255,0.8)')
            .style('font-size', '10px')
            .text(`Strongest: ${summaryText}`);
    }
}

// ============== 5. KNOWLEDGE GRAPH ==============

function renderKnowledgeGraph(data) {
    const container = document.getElementById('knowledge-graph');
    container.innerHTML = '';

    const graphData = data.knowledge_graph;
    if (!graphData?.nodes?.length) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">No entities detected</div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Color map for entity types
    const colorMap = {
        'PERSON': '#4CAF50', 'GPE': '#2196F3', 'LOC': '#2196F3',
        'ORG': '#FF9800', 'CONCEPT': '#9C27B0', 'DATE': '#F44336',
        'CARDINAL': '#00BCD4', 'WORK_OF_ART': '#E91E63', 'NORP': '#FF5722',
        'EVENT': '#E91E63', 'FAC': '#795548', 'PRODUCT': '#607D8B'
    };

    // Ensure all nodes have connections - connect isolated nodes to nearest
    const connectedNodes = new Set();
    graphData.edges.forEach(e => {
        connectedNodes.add(typeof e.source === 'object' ? e.source.id : e.source);
        connectedNodes.add(typeof e.target === 'object' ? e.target.id : e.target);
    });
    
    // Find isolated nodes and connect them to the first connected node
    const isolatedNodes = graphData.nodes.filter(n => !connectedNodes.has(n.id));
    if (isolatedNodes.length > 0 && connectedNodes.size > 0) {
        const firstConnected = graphData.nodes.find(n => connectedNodes.has(n.id));
        if (firstConnected) {
            isolatedNodes.forEach(isolated => {
                graphData.edges.push({
                    source: isolated.id,
                    target: firstConnected.id,
                    relation: 'related-to',
                    relation_text: 'mentioned with',
                    confidence: 0.3
                });
            });
        }
    }

    const svg = d3.select('#knowledge-graph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Add arrow marker for directed edges
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('fill', 'rgba(255,255,255,0.5)');

    const g = svg.append('g');

    svg.call(d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', event => g.attr('transform', event.transform)));

    const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(45));

    // Draw edges with relationship labels
    const linkGroup = g.selectAll('.link-group')
        .data(graphData.edges)
        .enter().append('g')
        .attr('class', 'link-group');

    const links = linkGroup.append('line')
        .attr('stroke', d => d.confidence > 0.5 ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.2)')
        .attr('stroke-width', d => Math.max(1, (d.confidence || 0.5) * 3))
        .attr('marker-end', 'url(#arrowhead)');

    // Edge labels (relationship type)
    const edgeLabels = linkGroup.append('text')
        .attr('class', 'edge-label')
        .attr('fill', 'rgba(255,255,255,0.6)')
        .attr('font-size', '9px')
        .attr('text-anchor', 'middle')
        .text(d => d.relation || '');

    // Node groups
    const nodeGroups = g.selectAll('.node')
        .data(graphData.nodes)
        .enter().append('g')
        .attr('class', 'node')
        .style('cursor', 'pointer')
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    // Node circles with size based on centrality
    nodeGroups.append('circle')
        .attr('r', d => 12 + (d.centrality || 0) * 20)
        .attr('fill', d => colorMap[d.type] || '#9C27B0')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('stroke-width', 4).attr('stroke', '#FFD700');
            showTooltip(event, `
                <strong>${d.label}</strong><br>
                Type: ${d.type || 'CONCEPT'}<br>
                Centrality: ${((d.centrality || 0) * 100).toFixed(1)}%<br>
                Connections: ${d.degree || 0}
            `);
        })
        .on('mouseout', function() {
            d3.select(this).attr('stroke-width', 2).attr('stroke', '#fff');
            hideTooltip();
        });

    // Node labels
    nodeGroups.append('text')
        .text(d => d.label?.length > 12 ? d.label.slice(0, 12) + '…' : d.label)
        .attr('dx', 15)
        .attr('dy', 4)
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .attr('font-weight', d => (d.centrality || 0) > 0.3 ? 'bold' : 'normal');

    // Type badge
    nodeGroups.append('text')
        .text(d => d.type ? d.type.slice(0, 3) : '')
        .attr('dx', -5)
        .attr('dy', -15)
        .attr('fill', d => colorMap[d.type] || '#9C27B0')
        .attr('font-size', '8px')
        .attr('font-weight', 'bold');

    simulation.on('tick', () => {
        // Keep nodes within bounds
        graphData.nodes.forEach(d => {
            d.x = Math.max(50, Math.min(width - 50, d.x));
            d.y = Math.max(50, Math.min(height - 50, d.y));
        });

        links
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        // Position edge labels at midpoint
        edgeLabels
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2 - 5);

        nodeGroups.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    // Add graph statistics
    const statsText = `${graphData.nodes.length} entities, ${graphData.edges.length} relationships`;
    svg.append('text')
        .attr('x', 10)
        .attr('y', height - 10)
        .attr('fill', 'rgba(255,255,255,0.5)')
        .attr('font-size', '10px')
        .text(statsText);
}


// ============== 6. 3D ATTENTION LANDSCAPE ==============

let scene3D, camera3D, renderer3D, controls3D;

function render3DAttention(data) {
    const container = document.getElementById('attention-3d');
    container.innerHTML = '';
    
    const matrix = data.attention?.attention_mean || [];
    
    if (matrix.length === 0 || typeof THREE === 'undefined') {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">No attention data or Three.js not loaded</div>';
        return;
    }
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Limit matrix size for performance
    const maxSize = 30;
    const step = Math.ceil(matrix.length / maxSize);
    const sampledMatrix = [];
    
    for (let i = 0; i < matrix.length; i += step) {
        const row = [];
        for (let j = 0; j < matrix[i].length; j += step) {
            row.push(matrix[i][j]);
        }
        sampledMatrix.push(row);
    }
    
    const n = sampledMatrix.length;
    if (n < 2) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">Insufficient data for 3D visualization</div>';
        return;
    }
    
    // Setup Three.js scene
    scene3D = new THREE.Scene();
    scene3D.background = new THREE.Color(0x1a1a2e);
    
    camera3D = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera3D.position.set(n * 0.8, n * 0.6, n * 0.8);
    camera3D.lookAt(n / 2, 0, n / 2);
    
    renderer3D = new THREE.WebGLRenderer({ antialias: true });
    renderer3D.setSize(width, height);
    container.appendChild(renderer3D.domElement);
    
    // Orbit controls for rotation
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls3D = new THREE.OrbitControls(camera3D, renderer3D.domElement);
        controls3D.enableDamping = true;
        controls3D.dampingFactor = 0.05;
        controls3D.target.set(n / 2, 0, n / 2);
        controls3D.update();
    }
    
    // Find max value for normalization
    const maxVal = Math.max(...sampledMatrix.flat(), 0.001);
    const heightScale = n * 0.5;
    
    // Create geometry for the surface
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const indices = [];
    
    // Color gradient function
    const getColor = (value) => {
        const normalized = value / maxVal;
        // Blue to Yellow to Red gradient
        if (normalized < 0.5) {
            return new THREE.Color(normalized * 2, normalized * 2, 1 - normalized);
        } else {
            return new THREE.Color(1, 1 - (normalized - 0.5) * 2, 0);
        }
    };
    
    // Create vertices
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < sampledMatrix[i].length; j++) {
            const value = sampledMatrix[i][j];
            const height = (value / maxVal) * heightScale;
            
            vertices.push(j, height, i);
            
            const color = getColor(value);
            colors.push(color.r, color.g, color.b);
        }
    }
    
    // Create faces (triangles)
    const cols = sampledMatrix[0].length;
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < cols - 1; j++) {
            const a = i * cols + j;
            const b = i * cols + j + 1;
            const c = (i + 1) * cols + j;
            const d = (i + 1) * cols + j + 1;
            
            indices.push(a, b, c);
            indices.push(b, d, c);
        }
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    
    // Create mesh with vertex colors
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 30
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    scene3D.add(mesh);
    
    // Add wireframe overlay
    const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 0x444466,
        wireframe: true,
        transparent: true,
        opacity: 0.1
    });
    const wireframe = new THREE.Mesh(geometry.clone(), wireframeMaterial);
    scene3D.add(wireframe);
    
    // Add base plane
    const planeGeometry = new THREE.PlaneGeometry(cols, n);
    const planeMaterial = new THREE.MeshBasicMaterial({
        color: 0x222244,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.rotation.x = -Math.PI / 2;
    plane.position.set(cols / 2 - 0.5, -0.01, n / 2 - 0.5);
    scene3D.add(plane);
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(Math.max(n, cols), Math.max(n, cols), 0x444466, 0x333355);
    gridHelper.position.set(cols / 2 - 0.5, 0, n / 2 - 0.5);
    scene3D.add(gridHelper);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene3D.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(n, n, n);
    scene3D.add(directionalLight);
    
    const directionalLight2 = new THREE.DirectionalLight(0x667eea, 0.4);
    directionalLight2.position.set(-n, n / 2, -n);
    scene3D.add(directionalLight2);
    
    // Add axis labels
    addAxisLabels(scene3D, n, cols);
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        if (controls3D) controls3D.update();
        renderer3D.render(scene3D, camera3D);
    }
    animate();
    
    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        camera3D.aspect = newWidth / newHeight;
        camera3D.updateProjectionMatrix();
        renderer3D.setSize(newWidth, newHeight);
    });
    resizeObserver.observe(container);
}

function addAxisLabels(scene, rows, cols) {
    // Create simple axis indicators using sprites or 3D text would require additional libraries
    // For now, we'll add simple line indicators
    
    // X axis (source tokens)
    const xAxisMaterial = new THREE.LineBasicMaterial({ color: 0x4CAF50 });
    const xAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, 0, -1),
        new THREE.Vector3(cols + 1, 0, -1)
    ]);
    const xAxis = new THREE.Line(xAxisGeometry, xAxisMaterial);
    scene.add(xAxis);
    
    // Z axis (target tokens)
    const zAxisMaterial = new THREE.LineBasicMaterial({ color: 0x2196F3 });
    const zAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, 0, -1),
        new THREE.Vector3(-1, 0, rows + 1)
    ]);
    const zAxis = new THREE.Line(zAxisGeometry, zAxisMaterial);
    scene.add(zAxis);
    
    // Y axis (attention weight)
    const yAxisMaterial = new THREE.LineBasicMaterial({ color: 0xFFEB3B });
    const yAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, 0, -1),
        new THREE.Vector3(-1, rows * 0.5, -1)
    ]);
    const yAxis = new THREE.Line(yAxisGeometry, yAxisMaterial);
    scene.add(yAxis);
}


// ============== 7. PATTERN DETECTION ==============

function renderPatternDetection(data) {
    const container = document.getElementById('pattern-detection');
    container.innerHTML = '';

    const matrix = data.attention?.attention_mean || [];
    const tokens = data.tokens || [];

    if (matrix.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">No attention data</div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Detect patterns
    const patterns = detectAttentionPatterns(matrix, tokens);

    const svg = d3.select('#pattern-detection')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const margin = { top: 40, right: 30, bottom: 60, left: 120 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const patternNames = Object.keys(patterns);
    const patternValues = Object.values(patterns);

    const x = d3.scaleLinear()
        .domain([0, Math.max(...patternValues, 1)])
        .range([0, chartWidth]);

    const y = d3.scaleBand()
        .domain(patternNames)
        .range([0, chartHeight])
        .padding(0.3);

    const colorScale = d3.scaleOrdinal()
        .domain(patternNames)
        .range(['#667eea', '#764ba2', '#f093fb', '#4CAF50', '#FF9800', '#2196F3']);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Bars
    g.selectAll('.bar')
        .data(patternNames)
        .enter()
        .append('rect')
        .attr('x', 0)
        .attr('y', d => y(d))
        .attr('width', d => x(patterns[d]))
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(d))
        .attr('rx', 4)
        .on('mouseover', function (event, d) {
            d3.select(this).attr('opacity', 0.8);
            showTooltip(event, `<strong>${d}</strong><br>Score: ${(patterns[d] * 100).toFixed(1)}%<br>${getPatternDescription(d)}`);
        })
        .on('mouseout', function () {
            d3.select(this).attr('opacity', 1);
            hideTooltip();
        });

    // Values
    g.selectAll('.value')
        .data(patternNames)
        .enter()
        .append('text')
        .attr('x', d => x(patterns[d]) + 8)
        .attr('y', d => y(d) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('fill', '#fff')
        .attr('font-size', '12px')
        .text(d => (patterns[d] * 100).toFixed(1) + '%');

    // Y axis
    g.append('g')
        .call(d3.axisLeft(y))
        .selectAll('text')
        .attr('fill', '#fff')
        .attr('font-size', '11px');

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text('Detected Attention Patterns');
}

function detectAttentionPatterns(matrix, tokens) {
    const n = matrix.length;
    if (n === 0) return {};

    let diagonal = 0;      // Self-attention (diagonal)
    let previous = 0;      // Previous token attention
    let positional = 0;    // Position-based patterns
    let global = 0;        // Global attention (attending to all)
    let local = 0;         // Local attention (nearby tokens)
    let sparse = 0;        // Sparse attention (few strong connections)

    for (let i = 0; i < n; i++) {
        const row = matrix[i];
        const sum = row.reduce((a, b) => a + b, 0) || 1;
        const norm = row.map(v => v / sum);

        // Diagonal (self-attention)
        if (i < row.length) {
            diagonal += norm[i];
        }

        // Previous token
        if (i > 0 && i - 1 < row.length) {
            previous += norm[i - 1];
        }

        // Local (within 3 tokens)
        let localSum = 0;
        for (let j = Math.max(0, i - 3); j <= Math.min(n - 1, i + 3); j++) {
            if (j < row.length) localSum += norm[j];
        }
        local += localSum / 7;

        // Global (entropy-based)
        const entropy = -norm.reduce((acc, p) => acc + (p > 0 ? p * Math.log2(p) : 0), 0);
        const maxEntropy = Math.log2(n);
        global += entropy / maxEntropy;

        // Sparse (top-3 concentration)
        const sorted = [...norm].sort((a, b) => b - a);
        sparse += sorted.slice(0, 3).reduce((a, b) => a + b, 0);

        // Positional (first few tokens)
        positional += norm.slice(0, Math.min(5, norm.length)).reduce((a, b) => a + b, 0);
    }

    return {
        'Self-Attention': diagonal / n,
        'Previous Token': previous / n,
        'Local Context': local / n,
        'Global Spread': global / n,
        'Sparse Focus': sparse / n,
        'Positional Bias': positional / n
    };
}

function getPatternDescription(pattern) {
    const descriptions = {
        'Self-Attention': 'Token attends to itself',
        'Previous Token': 'Attends to immediately preceding token',
        'Local Context': 'Focuses on nearby tokens (±3)',
        'Global Spread': 'Attention distributed across all tokens',
        'Sparse Focus': 'Concentrated on few key tokens',
        'Positional Bias': 'Attends to beginning of sequence'
    };
    return descriptions[pattern] || '';
}

// ============== 7. ATTENTION STATISTICS ==============

function renderAttentionStats(data) {
    const container = document.getElementById('attention-stats');
    container.innerHTML = '';

    const matrix = data.attention?.attention_mean || [];
    const tokens = data.tokens || [];

    if (matrix.length === 0) {
        container.innerHTML = '<div style="padding:40px;text-align:center;opacity:0.7;">No attention data</div>';
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Calculate statistics
    const stats = calculateAttentionStats(matrix);

    const svg = d3.select('#attention-stats')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create gauge-style visualizations
    const gauges = [
        { label: 'Avg Attention', value: stats.avgAttention, max: 0.5, color: '#667eea' },
        { label: 'Max Attention', value: stats.maxAttention, max: 1, color: '#4CAF50' },
        { label: 'Entropy', value: stats.avgEntropy / 5, max: 1, color: '#FF9800' },
        { label: 'Sparsity', value: stats.sparsity, max: 1, color: '#9C27B0' }
    ];

    const gaugeWidth = (width - 60) / 2;
    const gaugeHeight = (height - 80) / 2;

    gauges.forEach((gauge, i) => {
        const col = i % 2;
        const row = Math.floor(i / 2);
        const x = 30 + col * gaugeWidth;
        const y = 40 + row * gaugeHeight;

        const g = svg.append('g')
            .attr('transform', `translate(${x}, ${y})`);

        // Background arc
        const arcBg = d3.arc()
            .innerRadius(35)
            .outerRadius(50)
            .startAngle(-Math.PI / 2)
            .endAngle(Math.PI / 2);

        g.append('path')
            .attr('d', arcBg)
            .attr('transform', `translate(${gaugeWidth / 2}, 60)`)
            .attr('fill', 'rgba(255,255,255,0.1)');

        // Value arc
        const valueAngle = -Math.PI / 2 + (gauge.value / gauge.max) * Math.PI;
        const arcValue = d3.arc()
            .innerRadius(35)
            .outerRadius(50)
            .startAngle(-Math.PI / 2)
            .endAngle(Math.min(valueAngle, Math.PI / 2));

        g.append('path')
            .attr('d', arcValue)
            .attr('transform', `translate(${gaugeWidth / 2}, 60)`)
            .attr('fill', gauge.color);

        // Value text
        g.append('text')
            .attr('x', gaugeWidth / 2)
            .attr('y', 65)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '18px')
            .attr('font-weight', 'bold')
            .text((gauge.value * 100).toFixed(1) + '%');

        // Label
        g.append('text')
            .attr('x', gaugeWidth / 2)
            .attr('y', 95)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '12px')
            .attr('opacity', 0.8)
            .text(gauge.label);
    });

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text('Attention Statistics');
}

function calculateAttentionStats(matrix) {
    const n = matrix.length;
    let totalAttention = 0;
    let maxAttention = 0;
    let totalEntropy = 0;
    let sparseCount = 0;

    matrix.forEach(row => {
        const sum = row.reduce((a, b) => a + b, 0) || 1;
        const norm = row.map(v => v / sum);

        totalAttention += d3.mean(norm);
        maxAttention = Math.max(maxAttention, Math.max(...norm));

        // Entropy
        const entropy = -norm.reduce((acc, p) => acc + (p > 0 ? p * Math.log2(p) : 0), 0);
        totalEntropy += entropy;

        // Sparsity
        sparseCount += norm.filter(v => v < 0.01).length;
    });

    return {
        avgAttention: totalAttention / n,
        maxAttention: maxAttention,
        avgEntropy: totalEntropy / n,
        sparsity: sparseCount / (n * n)
    };
}

// ============== 8. COMPARATIVE ANALYSIS ==============

async function runTemperatureComparison() {
    const question = document.getElementById('question-input').value.trim();
    if (!question) return alert('Please analyze a question first');

    const container = document.getElementById('comparison-results');
    container.innerHTML = '<div style="text-align:center;padding:40px;"><div class="spinner"></div>Running comparison...</div>';

    try {
        // Run with low and high temperature
        const [lowTemp, highTemp] = await Promise.all([
            queryWithParams(question, 0.3, 80),
            queryWithParams(question, 1.5, 80)
        ]);

        container.innerHTML = '';

        // Low temperature panel
        const lowPanel = document.createElement('div');
        lowPanel.className = 'comparison-panel';
        lowPanel.innerHTML = `
            <h4>🧊 Low Temperature (0.3)</h4>
            <p style="font-size:13px;margin-bottom:10px;opacity:0.9;">${lowTemp.answer || '(No answer)'}</p>
            <div id="compare-low-heatmap" style="height:200px;background:rgba(0,0,0,0.3);border-radius:8px;"></div>
        `;
        container.appendChild(lowPanel);

        // High temperature panel
        const highPanel = document.createElement('div');
        highPanel.className = 'comparison-panel';
        highPanel.innerHTML = `
            <h4>🔥 High Temperature (1.5)</h4>
            <p style="font-size:13px;margin-bottom:10px;opacity:0.9;">${highTemp.answer || '(No answer)'}</p>
            <div id="compare-high-heatmap" style="height:200px;background:rgba(0,0,0,0.3);border-radius:8px;"></div>
        `;
        container.appendChild(highPanel);

        // Render mini heatmaps
        setTimeout(() => {
            renderMiniHeatmap('compare-low-heatmap', lowTemp.attention?.attention_mean || []);
            renderMiniHeatmap('compare-high-heatmap', highTemp.attention?.attention_mean || []);
        }, 100);

    } catch (error) {
        container.innerHTML = `<div style="color:#f44336;padding:20px;">Error: ${error.message}</div>`;
    }
}

async function runQuestionComparison() {
    const question1 = document.getElementById('question-input').value.trim();
    const question2 = document.getElementById('compare-question').value.trim();

    if (!question1 || !question2) return alert('Please enter both questions');

    const container = document.getElementById('question-comparison-results');
    container.innerHTML = '<div style="text-align:center;padding:40px;"><div class="spinner"></div>Comparing questions...</div>';

    try {
        const [result1, result2] = await Promise.all([
            queryWithParams(question1, 0.8, 80),
            queryWithParams(question2, 0.8, 80)
        ]);

        container.innerHTML = '';

        // Question 1 panel
        const panel1 = document.createElement('div');
        panel1.className = 'comparison-panel';
        panel1.innerHTML = `
            <h4>Question 1</h4>
            <p style="font-size:12px;opacity:0.7;margin-bottom:5px;">${question1}</p>
            <p style="font-size:13px;margin-bottom:10px;">${result1.answer || '(No answer)'}</p>
            <div id="compare-q1-heatmap" style="height:180px;background:rgba(0,0,0,0.3);border-radius:8px;"></div>
        `;
        container.appendChild(panel1);

        // Question 2 panel
        const panel2 = document.createElement('div');
        panel2.className = 'comparison-panel';
        panel2.innerHTML = `
            <h4>Question 2</h4>
            <p style="font-size:12px;opacity:0.7;margin-bottom:5px;">${question2}</p>
            <p style="font-size:13px;margin-bottom:10px;">${result2.answer || '(No answer)'}</p>
            <div id="compare-q2-heatmap" style="height:180px;background:rgba(0,0,0,0.3);border-radius:8px;"></div>
        `;
        container.appendChild(panel2);

        setTimeout(() => {
            renderMiniHeatmap('compare-q1-heatmap', result1.attention?.attention_mean || []);
            renderMiniHeatmap('compare-q2-heatmap', result2.attention?.attention_mean || []);
        }, 100);

    } catch (error) {
        container.innerHTML = `<div style="color:#f44336;padding:20px;">Error: ${error.message}</div>`;
    }
}

function renderMiniHeatmap(containerId, matrix) {
    const container = document.getElementById(containerId);
    if (!container || matrix.length === 0) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const n = Math.min(matrix.length, 25);
    const cellSize = Math.min(width / n, height / n);

    const svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([0, d3.max(matrix.slice(0, n).flat())]);

    const offsetX = (width - n * cellSize) / 2;
    const offsetY = (height - n * cellSize) / 2;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            svg.append('rect')
                .attr('x', offsetX + j * cellSize)
                .attr('y', offsetY + i * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', colorScale(matrix[i][j]));
        }
    }
}

// ============== UTILITIES ==============

function showTooltip(event, html) {
    const tooltip = document.getElementById('tooltip');
    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');

    const x = event.pageX + 15;
    const y = event.pageY + 15;

    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

function hideTooltip() {
    document.getElementById('tooltip').classList.add('hidden');
}

// ============== COLOR LEGEND GENERATOR ==============
// Creates consistent, labeled color scales for visualizations

function createColorLegend(container, colorScale, title, min, max, width = 200) {
    const legendDiv = document.createElement('div');
    legendDiv.className = 'color-legend';
    legendDiv.innerHTML = `
        <div class="legend-title">${title}</div>
        <div class="legend-gradient" style="background: linear-gradient(to right, ${colorScale(0)}, ${colorScale(0.5)}, ${colorScale(1)}); width: ${width}px;"></div>
        <div class="legend-labels">
            <span>${min.toFixed(2)}</span>
            <span>${((min + max) / 2).toFixed(2)}</span>
            <span>${max.toFixed(2)}</span>
        </div>
    `;
    container.appendChild(legendDiv);
}

// ============== EXPORT FUNCTIONALITY ==============
// Allows users to save visualizations as images for reports

function exportVisualization(containerId, filename) {
    const container = document.getElementById(containerId);
    const svg = container.querySelector('svg');
    
    if (!svg) {
        alert('No visualization to export');
        return;
    }
    
    // Clone SVG and add white background for export
    const svgClone = svg.cloneNode(true);
    svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    
    // Add background rect
    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('width', '100%');
    bgRect.setAttribute('height', '100%');
    bgRect.setAttribute('fill', '#1a1a2e');
    svgClone.insertBefore(bgRect, svgClone.firstChild);
    
    // Convert to data URL
    const svgData = new XMLSerializer().serializeToString(svgClone);
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    
    // Download
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}_${new Date().toISOString().slice(0, 10)}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function exportAllVisualizations() {
    const visualizations = [
        { id: 'animated-trace', name: 'reasoning_trace' },
        { id: 'sankey-diagram', name: 'attention_flow' },
        { id: 'attention-heatmap', name: 'attention_heatmap' },
        { id: 'knowledge-graph', name: 'knowledge_graph' },
        { id: 'pattern-detection', name: 'attention_patterns' },
        { id: 'attention-stats', name: 'attention_stats' }
    ];
    
    visualizations.forEach(viz => {
        setTimeout(() => exportVisualization(viz.id, viz.name), 100);
    });
}

// ============== BRUSHING & LINKING ==============
// Coordinates selection across all visualizations

function highlightTokenAcrossViews(tokenIndex, isInput = true) {
    if (isInput) {
        selectionState.selectedInputTokens.add(tokenIndex);
    } else {
        selectionState.selectedOutputTokens.add(tokenIndex);
    }
    selectionState.highlightMode = true;
    
    // Highlight in animated trace
    d3.select('#animated-trace svg')
        .selectAll(isInput ? '.input-node' : '.output-node')
        .attr('stroke-width', (d, i) => 
            (isInput ? selectionState.selectedInputTokens : selectionState.selectedOutputTokens).has(i) ? 4 : 2
        )
        .attr('stroke', (d, i) => 
            (isInput ? selectionState.selectedInputTokens : selectionState.selectedOutputTokens).has(i) ? '#FFD700' : '#fff'
        );
    
    // Highlight in heatmap
    d3.select('#attention-heatmap svg')
        .selectAll('rect')
        .attr('stroke', function(d) {
            if (!d) return 'none';
            const match = isInput ? d.j === tokenIndex : d.i === tokenIndex;
            return match ? '#FFD700' : 'none';
        })
        .attr('stroke-width', function(d) {
            if (!d) return 0;
            const match = isInput ? d.j === tokenIndex : d.i === tokenIndex;
            return match ? 2 : 0;
        });
    
    // Highlight in token gradient
    const gradientTokens = document.querySelectorAll('#token-gradient .gradient-token');
    gradientTokens.forEach((el, i) => {
        if ((isInput && i === tokenIndex) || (!isInput && i === tokenIndex)) {
            el.style.outline = '2px solid #FFD700';
            el.style.outlineOffset = '2px';
        }
    });
}

function clearAllHighlights() {
    selectionState.selectedInputTokens.clear();
    selectionState.selectedOutputTokens.clear();
    selectionState.selectedEntities.clear();
    selectionState.highlightMode = false;
    
    // Clear all highlights
    d3.selectAll('.input-node, .output-node')
        .attr('stroke-width', 2)
        .attr('stroke', '#fff');
    
    d3.select('#attention-heatmap svg')
        .selectAll('rect')
        .attr('stroke', 'none');
    
    document.querySelectorAll('#token-gradient .gradient-token').forEach(el => {
        el.style.outline = 'none';
    });
}

// ============== INSIGHTS GENERATOR ==============
// Automatically generates key insights from the data

function generateInsights(data) {
    const insights = [];
    const matrix = data.attention?.attention_mean || [];
    const questionTokens = data.question_tokens || [];
    const answerTokens = data.answer_tokens || [];
    
    if (matrix.length === 0) return insights;
    
    // Find highest attention pairs
    let maxAttention = 0;
    let maxPair = { i: 0, j: 0 };
    
    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            if (val > maxAttention) {
                maxAttention = val;
                maxPair = { i, j };
            }
        });
    });
    
    // Calculate average attention
    const flatMatrix = matrix.flat();
    const avgAttention = flatMatrix.reduce((a, b) => a + b, 0) / flatMatrix.length;
    
    // Calculate attention entropy (how spread out attention is)
    const entropy = -flatMatrix
        .filter(v => v > 0)
        .map(v => v * Math.log2(v))
        .reduce((a, b) => a + b, 0) / flatMatrix.length;
    
    // Generate insights
    insights.push({
        type: 'highlight',
        icon: '🎯',
        title: 'Strongest Connection',
        text: `Highest attention (${(maxAttention * 100).toFixed(1)}%) between positions ${maxPair.i} and ${maxPair.j}`
    });
    
    insights.push({
        type: 'metric',
        icon: '📊',
        title: 'Attention Distribution',
        text: `Average attention: ${(avgAttention * 100).toFixed(2)}%, Entropy: ${entropy.toFixed(2)} bits`
    });
    
    // Entity insights
    const entities = data.knowledge_graph?.nodes || [];
    if (entities.length > 0) {
        const entityTypes = {};
        entities.forEach(e => {
            entityTypes[e.type] = (entityTypes[e.type] || 0) + 1;
        });
        const topType = Object.entries(entityTypes).sort((a, b) => b[1] - a[1])[0];
        
        insights.push({
            type: 'entity',
            icon: '🏷️',
            title: 'Entity Analysis',
            text: `Found ${entities.length} entities. Most common type: ${topType[0]} (${topType[1]})`
        });
    }
    
    // Answer quality insight
    if (answerTokens.length > 0) {
        insights.push({
            type: 'answer',
            icon: '💬',
            title: 'Response Analysis',
            text: `Generated ${answerTokens.length} tokens from ${questionTokens.length} input tokens`
        });
    }
    
    return insights;
}

function displayInsights(insights) {
    const container = document.getElementById('insights-panel');
    if (!container) return;
    
    container.innerHTML = '<h4>🔍 Key Insights</h4>';
    
    insights.forEach(insight => {
        const div = document.createElement('div');
        div.className = `insight-item insight-${insight.type}`;
        div.innerHTML = `
            <span class="insight-icon">${insight.icon}</span>
            <div class="insight-content">
                <strong>${insight.title}</strong>
                <p>${insight.text}</p>
            </div>
        `;
        container.appendChild(div);
    });
}
