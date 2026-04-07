/**
 * Sports Analytics System Dashboard Logic
 */

// Global Chart Options
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';

// Chart instances
let speedChartObj = null;
let riskChartObj = null;
let jointChartObj = null;
let valgusChartObj = null;
let trunkChartObj = null;

// API Base URL (adjust if running on different port)
const API_BASE = window.location.origin;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const jobId = urlParams.get('job_id');
    
    if (jobId) {
        loadByJobId(jobId);
    } else {
        loadLatest();
    }
});

// --- Navigation & UI ---

async function toggleHistory() {
    const section = document.getElementById('historySection');
    if (section.style.display === 'none') {
        section.style.display = 'block';
        await fetchHistory();
    } else {
        section.style.display = 'none';
    }
}

function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('collapsed');
}

// --- Data Fetching ---

async function loadByJobId(jobId) {
    try {
        const response = await fetch(`${API_BASE}/analyses/${jobId}`);
        if (response.ok) {
            const analysis = await response.json();
            displayAnalysis(analysis);
        } else {
            console.error(`Analysis ${jobId} not found.`);
            loadLatest();
        }
    } catch (error) {
        console.error('Error fetching analysis:', error);
        loadLatest();
    }
}

async function loadLatest() {
    try {
        const response = await fetch(`${API_BASE}/analyses/latest`);
        if (response.ok) {
            const analysis = await response.json();
            displayAnalysis(analysis);
        } else {
            console.log("No previous analyses found.");
        }
    } catch (error) {
        console.error('Error fetching latest:', error);
    }
}

async function fetchHistory() {
    const list = document.getElementById('historyList');
    try {
        const response = await fetch(`${API_BASE}/analyses`);
        const data = await response.json();
        
        list.innerHTML = '';
        if (data.length === 0) {
            list.innerHTML = '<p class="meta">No history found</p>';
            return;
        }

        data.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            const date = new Date(item.created_at).toLocaleString();
            div.innerHTML = `
                <h4>Player #${item.player_id} - ${item.session_tags || 'Quick Scan'}</h4>
                <div class="meta">${date} | ${item.yolo_size.toUpperCase()} Tracking</div>
            `;
            div.onclick = () => {
                displayAnalysis(item);
                toggleHistory();
            };
            list.appendChild(div);
        });
    } catch (error) {
        list.innerHTML = '<p class="meta">Error loading history</p>';
    }
}

function displayAnalysis(analysis) {
    if (!analysis) return;

    // Update Header
    document.getElementById('viewTitle').textContent = `Analysis: Player #${analysis.player_id}`;
    document.getElementById('sessionInfo').textContent = `${analysis.session_tags || 'Baseline Scan'} | ${new Date(analysis.created_at).toLocaleDateString()}`;

    // Update Video (Adopted from Old DASH / Native compatibility)
    const video = document.getElementById('analysisVideo');
    
    // Add error listener to help debug codec issues
    video.onerror = () => {
        const err = video.error;
        console.error("Video Playback Error:", err);
        if (err && err.code === 4) {
            alert("Video Codec Error: Your browser cannot play this video format. Try a different browser like Chrome or Edge.");
        }
    };

    if (analysis.video_url) {
        console.log("Applying video source:", analysis.video_url);
        
        // Reset video state
        video.pause();
        
        // Update both the video src and the source element for maximum compatibility
        const source = video.querySelector('source');
        if (source) {
            source.src = analysis.video_url;
            source.type = 'video/mp4'; 
        }
        video.src = analysis.video_url; 
        
        // Ensure muted autoplay (standard for dashboard visuals)
        video.muted = true; 
        video.load();
        
        // Slight delay for source internal buffering
        setTimeout(() => {
            const playPromise = video.play();
            if (playPromise !== undefined) {
                playPromise.catch(e => {
                    console.warn("Auto-play blocked or failed:", e);
                    // If it failed because of user interaction required, it's okay
                });
            }
        }, 150);
    }

    // --- NEW: Sync URL with the ID so users can share direct links ---
    if (analysis.id) {
        const url = new URL(window.location);
        url.searchParams.set('job_id', analysis.id);
        window.history.replaceState({}, '', url);
    }

    // Update KPIs and Charts
    updateSummary(analysis.summary);
    
    // Update Resources
    populateResources(analysis);
    
    if (analysis.summary && analysis.summary.frame_metrics) {
        renderCharts(analysis.summary.frame_metrics);
    } else if (analysis.data_urls && analysis.data_urls['analytics_unified.json']) {
        // Fallback: fetch from public URL
        fetch(analysis.data_urls['analytics_unified.json'])
            .then(res => res.json())
            .then(data => renderCharts(data.frame_metrics));
    }
}

function populateResources(analysis) {
    const dataList = document.getElementById('dataFilesList');
    const plotList = document.getElementById('plotFilesList');

    dataList.innerHTML = '';
    plotList.innerHTML = '';

    // Data Files
    if (analysis.data_urls) {
        Object.entries(analysis.data_urls).forEach(([name, url]) => {
            const ext = name.split('.').pop().toUpperCase();
            dataList.appendChild(createFileLink(name, url, ext));
        });
    }

    // Static Plots
    if (analysis.plot_urls) {
        Object.entries(analysis.plot_urls).forEach(([name, url]) => {
            const ext = name.split('.').pop().toUpperCase();
            plotList.appendChild(createFileLink(name, url, ext));
        });
    }
}

function createFileLink(name, url, type) {
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.className = 'file-link';
    a.innerHTML = `
        <span>${name}</span>
        <span class="badge type-icon">${type}</span>
    `;
    return a;
}

// --- Upload & Analysis ---


// --- Chart Rendering (adapted from original) ---

function updateSummary(summary) {
    if (!summary || !summary.player_summary) return;
    const s = summary.player_summary;

    // Peak Risk
    const peakRisk = s.peak_risk_score.toFixed(1);
    const riskLabel = s.fall_risk_label || 'Low';
    document.getElementById('kpiRisk').textContent = `${peakRisk}/100`;
    document.getElementById('kpiRiskLabel').textContent = `Overall Risk: ${riskLabel}`;
    
    const riskCard = document.getElementById('riskCard');
    riskCard.className = 'kpi-card glass';
    if (riskLabel.toLowerCase() === 'high') riskCard.classList.add('risk-high');
    else if (riskLabel.toLowerCase() === 'medium') riskCard.classList.add('risk-medium');
    else riskCard.classList.add('risk-low');

    // Fatigue
    document.getElementById('kpiFatigue').textContent = s.fatigue_label || 'Low';
    const fCard = document.getElementById('fatigueCard');
    fCard.className = 'kpi-card glass'; // Reset
    if (s.fatigue_label === 'High') fCard.classList.add('risk-high');
    else if (s.fatigue_label === 'Medium') fCard.classList.add('risk-medium');
    else fCard.classList.add('risk-low');

    // Injury
    document.getElementById('kpiInjury').textContent = s.injury_risk_label || 'Normal';
    document.getElementById('kpiInjuryDetail').textContent = s.injury_risk_detail || 'No anomalies';
    const iCard = document.getElementById('injuryCard');
    iCard.className = 'kpi-card glass'; // Reset
    if (s.injury_risk_label === 'High') iCard.classList.add('risk-high');
    else iCard.classList.add('risk-low');

    document.getElementById('kpiSpeed').textContent = `${s.max_speed.toFixed(2)} m/s`;
    document.getElementById('kpiSpeedLabel').textContent = `Avg: ${s.avg_speed.toFixed(2)} m/s`;

    // Detailed Biometrics Calculation (Restore from Old Dash logic)
    const metrics = summary.frame_metrics || [];
    if (metrics.length > 0) {
        const avg = (key) => {
            const vals = metrics.map(m => m[key] || m[`bio_${key}`]).filter(v => v !== undefined && !isNaN(v));
            return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
        };

        const setVal = (id, val, unit = '') => {
            const el = document.getElementById(id);
            if (el) el.textContent = `${val.toFixed(id.includes('Width') ? 2 : 1)}${unit}`;
        };

        setVal('valStepWidth', s.avg_stride_length * 0.12, ' m'); 
        setVal('valTrunkLean', Math.abs(avg('trunk_lean')), '°');
        setVal('valDoubleSupport', s.double_support_pct || 0, '%'); 
        setVal('valPelvicRot', Math.abs(s.avg_pelvic_rotation || 0), '°');
        setVal('valTrunkSag', Math.abs(avg('bio_trunk_sagittal_lean')), '°');
        setVal('valArmSwing', avg('bio_left_arm_swing'), '°');
    }

    document.getElementById('kpiEnergy').textContent = `${s.estimated_energy_kcal_hr.toFixed(0)} kcal/hr`;
    document.getElementById('kpiDistance').textContent = `Dist: ${s.total_distance_m.toFixed(1)} m`;

    document.getElementById('kpiSymmetry').textContent = `${s.gait_symmetry_pct.toFixed(1)} %`;
    document.getElementById('kpiStride').textContent = `Stride: ${s.avg_stride_length.toFixed(2)} m`;
}

function renderCharts(frames) {
    if (!frames || frames.length === 0) return;

    // Destroy old charts to prevent overlapping
    if (speedChartObj) speedChartObj.destroy();
    if (riskChartObj) riskChartObj.destroy();
    if (jointChartObj) jointChartObj.destroy();
    if (valgusChartObj) valgusChartObj.destroy();
    if (trunkChartObj) trunkChartObj.destroy();

    const step = Math.max(1, Math.floor(frames.length / 150));
    const sampledFrames = frames.filter((_, i) => i % step === 0);
    const labels = sampledFrames.map(f => f.timestamp.toFixed(2) + 's');
    
    // --- Speed Chart ---
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    const speedGradient = speedCtx.createLinearGradient(0, 0, 0, 400);
    speedGradient.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    speedGradient.addColorStop(1, 'rgba(0, 240, 255, 0)');

    speedChartObj = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Speed (m/s)',
                data: sampledFrames.map(f => f.speed),
                borderColor: '#00f0ff',
                backgroundColor: speedGradient,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });

    // --- Valgus Chart (New) ---
    const valgusCtx = document.getElementById('valgusChart').getContext('2d');
    valgusChartObj = new Chart(valgusCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'L. Valgus',
                data: sampledFrames.map(f => f.l_valgus_clinical),
                borderColor: '#ef4444',
                tension: 0.4,
                pointRadius: 0
            }, {
                label: 'R. Valgus',
                data: sampledFrames.map(f => f.r_valgus_clinical),
                borderColor: '#fbbf24',
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // --- Risk Chart ---
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    riskChartObj = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Score (%)',
                data: sampledFrames.map(f => f.risk_score),
                borderColor: '#ff00ff',
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
    });

    // --- Joint Chart ---
    const jointCtx = document.getElementById('jointChart').getContext('2d');
    jointChartObj = new Chart(jointCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'L. Knee Angle',
                data: sampledFrames.map(f => f.left_knee_angle),
                borderColor: '#10b981',
                borderWidth: 2,
                pointRadius: 0
            }, {
                label: 'R. Knee Angle',
                data: sampledFrames.map(f => f.right_knee_angle),
                borderColor: '#8b5cf6',
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // --- Trunk Chart (New) ---
    const trunkCtx = document.getElementById('trunkChart').getContext('2d');
    trunkChartObj = new Chart(trunkCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trunk Lean (Deg)',
                data: sampledFrames.map(f => f.trunk_lean),
                borderColor: '#60a5fa',
                backgroundColor: 'rgba(96, 165, 250, 0.2)',
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}
