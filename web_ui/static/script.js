// Frontend JavaScript for DeepSeek Dataset Generator & Trainer

let currentFilter = 'all';
let refreshInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Setup form handlers
    setupDatasetForm();
    setupTrainingForm();

    // Load jobs initially
    refreshJobs();

    // Auto-refresh jobs every 5 seconds when on jobs tab
    const jobsTab = document.getElementById('jobs-tab');
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.target.classList.contains('active')) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });
    });
    observer.observe(jobsTab, { attributes: true, attributeFilter: ['class'] });
});

// Tab Switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    // Refresh jobs if switching to jobs tab
    if (tabName === 'jobs') {
        refreshJobs();
    }
}

// Dataset Form
function setupDatasetForm() {
    const form = document.getElementById('dataset-form');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = {
            source_dataset: document.getElementById('source_dataset').value,
            source_subset: document.getElementById('source_subset').value || null,
            source_split: document.getElementById('source_split').value,
            text_field: document.getElementById('text_field').value,
            summary_field: document.getElementById('summary_field').value,
            output_dir: document.getElementById('output_dir').value,
            max_samples: document.getElementById('max_samples').value ?
                         parseInt(document.getElementById('max_samples').value) : null,
            hf_username: document.getElementById('hf_username').value || null,
            dataset_name: document.getElementById('dataset_name').value || null,
            private_dataset: document.getElementById('private_dataset').checked
        };

        try {
            const response = await fetch('/api/dataset/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                showToast('Dataset generation started! Job ID: ' + data.job_id, 'success');
                // Switch to jobs tab
                switchTab('jobs');
            } else {
                showToast('Error: ' + data.error, 'error');
            }
        } catch (error) {
            showToast('Error starting dataset generation: ' + error.message, 'error');
        }
    });
}

// Training Form
function setupTrainingForm() {
    const form = document.getElementById('training-form');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = {
            dataset_repo: document.getElementById('dataset_repo').value,
            model_name: document.getElementById('model_name').value,
            output_dir: document.getElementById('output_dir_train').value,
            batch_size: parseInt(document.getElementById('batch_size').value),
            num_epochs: parseInt(document.getElementById('num_epochs').value),
            learning_rate: document.getElementById('learning_rate').value,
            max_length: parseInt(document.getElementById('max_length').value),
            mixed_precision: document.getElementById('mixed_precision').value,
            gradient_accumulation_steps: parseInt(document.getElementById('gradient_accumulation_steps').value),
            push_to_hub: document.getElementById('push_to_hub').checked,
            hub_model_id: document.getElementById('hub_model_id').value || null
        };

        try {
            const response = await fetch('/api/model/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                showToast('Model training started! Job ID: ' + data.job_id, 'success');
                // Switch to jobs tab
                switchTab('jobs');
            } else {
                showToast('Error: ' + data.error, 'error');
            }
        } catch (error) {
            showToast('Error starting training: ' + error.message, 'error');
        }
    });
}

// Jobs Management
async function refreshJobs() {
    try {
        const url = currentFilter === 'all' ?
                    '/api/jobs' :
                    `/api/jobs?type=${currentFilter}`;

        const response = await fetch(url);
        const data = await response.json();

        displayJobs(data.jobs);
    } catch (error) {
        console.error('Error fetching jobs:', error);
        showToast('Error loading jobs', 'error');
    }
}

function filterJobs(filter) {
    currentFilter = filter;
    refreshJobs();
}

function displayJobs(jobs) {
    const container = document.getElementById('jobs-container');

    if (!jobs || jobs.length === 0) {
        container.innerHTML = '<p class="no-jobs">No jobs found</p>';
        return;
    }

    container.innerHTML = jobs.map(job => createJobCard(job)).join('');
}

function createJobCard(job) {
    const progress = job.progress;
    const statusClass = getStatusClass(progress.status);
    const progressPercent = progress.progress_percentage || 0;

    return `
        <div class="job-card">
            <div class="job-header">
                <div>
                    <span class="job-type">${formatJobType(job.job_type)}</span>
                    <span class="job-id">${job.job_id}</span>
                </div>
                <span class="status-badge ${statusClass}">${progress.status}</span>
            </div>

            <div class="progress-bar">
                <div class="progress-fill" style="width: ${progressPercent}%"></div>
            </div>

            <div class="job-stats">
                <div class="stat">
                    <span class="stat-label">Progress:</span>
                    <span class="stat-value">${progressPercent.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Processed:</span>
                    <span class="stat-value">${progress.processed_samples} / ${progress.total_samples}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Unique:</span>
                    <span class="stat-value">${progress.unique_samples}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Failed:</span>
                    <span class="stat-value">${progress.failed_samples}</span>
                </div>
            </div>

            ${progress.last_error ? `
                <div class="error-message">
                    <strong>Last Error:</strong> ${progress.last_error}
                </div>
            ` : ''}

            <div class="job-actions">
                ${getJobActions(job.job_id, progress.status)}
            </div>

            <div class="job-footer">
                <span>Created: ${formatDate(job.created_at)}</span>
                <span>Updated: ${formatDate(job.updated_at)}</span>
            </div>
        </div>
    `;
}

function getJobActions(jobId, status) {
    let actions = '';

    if (status === 'in_progress') {
        actions += `<button class="btn btn-small btn-warning" onclick="pauseJob('${jobId}')">Pause</button>`;
    }

    if (status === 'paused' || status === 'failed') {
        actions += `<button class="btn btn-small btn-primary" onclick="resumeJob('${jobId}')">Resume</button>`;
    }

    actions += `<button class="btn btn-small btn-secondary" onclick="viewJobDetails('${jobId}')">Details</button>`;
    actions += `<button class="btn btn-small btn-danger" onclick="deleteJob('${jobId}')">Delete</button>`;

    return actions;
}

async function pauseJob(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/pause`, { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            showToast('Job paused', 'success');
            refreshJobs();
        } else {
            showToast('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error pausing job', 'error');
    }
}

async function resumeJob(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/resume`, { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            showToast('Job resumed', 'success');
            refreshJobs();
        } else {
            showToast('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error resuming job', 'error');
    }
}

async function viewJobDetails(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const data = await response.json();

        if (response.ok) {
            alert(JSON.stringify(data, null, 2));
        } else {
            showToast('Error loading job details', 'error');
        }
    } catch (error) {
        showToast('Error loading job details', 'error');
    }
}

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job?')) {
        return;
    }

    try {
        const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        const data = await response.json();

        if (response.ok) {
            showToast('Job deleted', 'success');
            refreshJobs();
        } else {
            showToast('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error deleting job', 'error');
    }
}

// Utility Functions
function getStatusClass(status) {
    const classes = {
        'pending': 'status-pending',
        'in_progress': 'status-in-progress',
        'completed': 'status-completed',
        'failed': 'status-failed',
        'paused': 'status-paused'
    };
    return classes[status] || '';
}

function formatJobType(type) {
    const types = {
        'dataset_generation': 'Dataset Generation',
        'model_training': 'Model Training'
    };
    return types[type] || type;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 5000);
}

function startAutoRefresh() {
    if (!refreshInterval) {
        refreshInterval = setInterval(refreshJobs, 5000);
    }
}

function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}
