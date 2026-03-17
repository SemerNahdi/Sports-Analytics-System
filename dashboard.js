/**
 * Juventus Sports Analytics Dashboard Logic
 */

// Global Chart Options for aesthetics
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.scale.grid.borderColor = 'transparent';

// Chart instances
let speedChartObj = null;
let riskChartObj = null;
let jointChartObj = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});

async function loadData() {
    try {
        // Fetch the metrics JSON file from the Output directory
        const response = await fetch('Output/metrics.json');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update UI
        updateSummary(data.player_summary);
        renderCharts(data.frame_metrics);
        
    } catch (error) {
        console.error('Failed to load metrics.json:', error);
        document.getElementById('sessionInfo').textContent = 'Error loading data. Make sure you are running via a local server (e.g. python serve_dashboard.py).';
    }
}

function updateSummary(summary) {
    if (!summary) return;

    document.getElementById('sessionInfo').textContent = `Duration: ${summary.duration_seconds.toFixed(2)}s | Total Frames: ${summary.total_frames}`;

    // Peak Risk
    const peakRisk = summary.peak_risk_score.toFixed(1);
    const riskLabel = summary.fall_risk_label || 'Unknown';
    document.getElementById('kpiRisk').textContent = `${peakRisk}/100`;
    document.getElementById('kpiRiskLabel').textContent = `Overall Risk: ${riskLabel}`;
    
    // Update card styling based on risk label
    const riskCard = document.getElementById('riskCard');
    riskCard.className = 'kpi-card glass'; // reset
    if (riskLabel.toLowerCase() === 'high') riskCard.classList.add('risk-high');
    else if (riskLabel.toLowerCase() === 'medium') riskCard.classList.add('risk-medium');
    else riskCard.classList.add('risk-low');

    // Speed
    document.getElementById('kpiSpeed').textContent = `${summary.max_speed.toFixed(2)} m/s`;
    document.getElementById('kpiSpeedLabel').textContent = `Avg: ${summary.avg_speed.toFixed(2)} m/s`;

    // Energy & Distance
    document.getElementById('kpiEnergy').textContent = `${summary.estimated_energy_kcal_hr.toFixed(0)} kcal/hr`;
    document.getElementById('kpiDistance').textContent = `Dist: ${summary.total_distance_m.toFixed(1)} m`;

    // Gait & Cadence
    document.getElementById('kpiSymmetry').textContent = `${summary.gait_symmetry_pct.toFixed(1)} %`;
    document.getElementById('kpiCadence').textContent = `Cadence: ${summary.avg_cadence.toFixed(0)} spm`;
}

function renderCharts(frames) {
    if (!frames || frames.length === 0) return;

    // We only take a subset to avoid overwhelming the chart, or take all if small
    // Here we'll take every Nth frame if > 300, else take all.
    const step = Math.max(1, Math.floor(frames.length / 150));
    const sampledFrames = frames.filter((_, i) => i % step === 0);

    const labels = sampledFrames.map(f => f.timestamp.toFixed(2) + 's');
    
    // Create Gradients
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    const speedGradient = speedCtx.createLinearGradient(0, 0, 0, 400);
    speedGradient.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
    speedGradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const riskGradient = riskCtx.createLinearGradient(0, 0, 0, 400);
    riskGradient.addColorStop(0, 'rgba(239, 68, 68, 0.5)'); // red
    riskGradient.addColorStop(1, 'rgba(239, 68, 68, 0.0)');
    
    const injuryGradient = riskCtx.createLinearGradient(0, 0, 0, 400);
    injuryGradient.addColorStop(0, 'rgba(245, 158, 11, 0.5)'); // orange
    injuryGradient.addColorStop(1, 'rgba(245, 158, 11, 0.0)');


    // ------ CHART 1: Speed & Acceleration ------
    speedChartObj = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Speed (m/s)',
                data: sampledFrames.map(f => f.speed),
                borderColor: '#00f0ff',
                backgroundColor: speedGradient,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                yAxisID: 'y'
            }, {
                label: 'Acceleration (m/s²)',
                data: sampledFrames.map(f => f.acceleration),
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderDash: [5, 5],
                borderWidth: 1,
                tension: 0.1,
                fill: false,
                pointRadius: 0,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', align: 'end' },
                tooltip: { 
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleFont: { size: 13 },
                    bodyFont: { size: 13 },
                    padding: 10,
                    cornerRadius: 8
                }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Speed' }, beginAtZero: true },
                y1: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Accel' } }
            }
        }
    });


    // ------ CHART 2: Risk Probabilities ------
    // Fall risk, Injury risk
    riskChartObj = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fall Risk Probability',
                data: sampledFrames.map(f => f.fall_risk * 100),
                borderColor: '#ef4444',
                backgroundColor: riskGradient,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }, {
                label: 'Injury Risk Probability',
                data: sampledFrames.map(f => f.injury_risk * 100),
                borderColor: '#f59e0b',
                backgroundColor: injuryGradient,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', align: 'end' }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Probability (%)' }, beginAtZero: true, max: 100 }
            }
        }
    });

    // ------ CHART 3: Joint Angles ------
    const jointCtx = document.getElementById('jointChart').getContext('2d');
    jointChartObj = new Chart(jointCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Left Knee Angle',
                data: sampledFrames.map(f => f.left_knee_angle),
                borderColor: '#10b981', // green
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 0
            }, {
                label: 'Right Knee Angle',
                data: sampledFrames.map(f => f.right_knee_angle),
                borderColor: '#8b5cf6', // purple
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', align: 'end' }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Angle (degrees)' } }
            }
        }
    });

}
