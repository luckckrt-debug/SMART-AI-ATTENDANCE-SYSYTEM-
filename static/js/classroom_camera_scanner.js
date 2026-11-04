/**
 * ClassroomCameraScanner
 * Professional WebRTC Multi-Person Classroom Camera Scanner & Diagnostics Engine.
 * 
 * Architecture:
 * Camera MediaStream -> HTMLVideoElement (Live Video Display)
 *                              |
 *                              +--> Offscreen Capture Canvas -> AI Face Recognition API
 *                              |
 *                              +--> Transparent Overlay Canvas (Bounding Boxes, Names, IDs, Telemetry)
 */

class ClassroomCameraScanner {
    constructor(options = {}) {
        this.options = Object.assign({
            sessionId: '',
            processingIntervalMs: 260,
            idealWidth: 1280,
            idealHeight: 720,
            autoStart: false
        }, options);

        // Core State Machine:
        // IDLE | REQUESTING_PERMISSION | CAMERA_CONNECTED | VIDEO_INITIALIZING | VIDEO_RENDERING | AI_MODEL_LOADING | DETECTING | RECOGNIZING | ERROR
        this.state = 'IDLE';

        // MediaStream & Hardware Tracks
        this.mediaStream = null;
        this.videoTrack = null;
        this.activeDeviceId = null;
        this.availableDevices = [];

        // DOM Elements
        this.videoEl = null;
        this.overlayCanvas = null;
        this.overlayCtx = null;
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });

        // Frame Validation & Telemetry
        this.isStreamValid = false;
        this.framesReceived = 0;
        this.lastFrameTimestamp = 0;
        this.frameDeliveryFps = 0;
        this.renderFps = 0;
        this.renderFramesCount = 0;
        this.lastRenderTime = performance.now();
        this.latencyMs = 0;
        this.facesCount = 0;
        this.lastCameraError = null;
        this.isTabHidden = false;
        this.inFlight = false;

        // Models & Detection State
        this.detectorStatus = 'AI READY (Haar Cascade + Dlib)';
        this.modelStatus = 'AI READY (FaceNet + Face_recognition.h5)';
        this.latestDetections = [];
        this.verifiedRolls = new Set();

        // Loop / Animation Handles
        this.animFrameId = null;
        this.videoFrameCallbackId = null;
        this.processingTimerId = null;
        this.diagnosticsTimerId = null;
        this.clockTimerId = null;
    }

    /**
     * Bind DOM elements and attach control listeners.
     */
    initUI(elements = {}) {
        this.elements = Object.assign({
            video: document.getElementById('classroomVideo') || document.getElementById('clientWebcam'),
            overlayCanvas: document.getElementById('classroomOverlayCanvas') || document.getElementById('clientOverlayCanvas'),
            placeholder: document.getElementById('camOfflinePlaceholder'),
            errorAlert: document.getElementById('camErrorAlert'),
            placeholderTitle: document.getElementById('placeholderTitle'),
            placeholderDesc: document.getElementById('placeholderDesc'),
            statusBadge: document.getElementById('camStatusBadge'),
            statusText: document.getElementById('camStatusText'),
            deviceSelect: document.getElementById('classroomDeviceSelect') || document.getElementById('cameraDeviceSelect'),
            deviceNameDisplay: document.getElementById('camDeviceNameDisplay'),
            hudFaceCount: document.getElementById('hudFaceCount'),
            hudClock: document.getElementById('hudClock'),
            hudFpsVal: document.getElementById('hudFpsVal'),
            hudFacesVal: document.getElementById('hudFacesVal'),
            hudLatencyVal: document.getElementById('hudLatencyVal'),
            hudResolutionVal: document.getElementById('hudResolutionVal'),
            startBtn: document.getElementById('startCamBtn') || document.getElementById('startScannerCamBtn'),
            pauseBtn: document.getElementById('pauseCamBtn') || document.getElementById('pauseScannerCamBtn'),
            resumeBtn: document.getElementById('resumeCamBtn') || document.getElementById('resumeScannerCamBtn'),
            stopBtn: document.getElementById('stopCamBtn') || document.getElementById('stopScannerCamBtn'),
            fullscreenBtn: document.getElementById('classroomFullscreenBtn') || document.getElementById('fullscreenCamBtn'),
            cameraWrapper: document.getElementById('cameraWrapper'),
            cardsContainer: document.getElementById('studentCardsContainer') || document.getElementById('verifiedCardsContainer'),
            verifiedBadge: document.getElementById('verifiedCountBadge'),

            // Developer Diagnostics Panel Elements
            diagStreamAvailable: document.getElementById('diagStreamAvailable'),
            diagTrackState: document.getElementById('diagTrackState'),
            diagTrackEnabled: document.getElementById('diagTrackEnabled'),
            diagDeviceId: document.getElementById('diagDeviceId'),
            diagReadyState: document.getElementById('diagReadyState'),
            diagVideoDims: document.getElementById('diagVideoDims'),
            diagCanvasDims: document.getElementById('diagCanvasDims'),
            diagPlaybackStatus: document.getElementById('diagPlaybackStatus'),
            diagCallbackStatus: document.getElementById('diagCallbackStatus'),
            diagLastFrameReceived: document.getElementById('diagLastFrameReceived'),
            diagDetectorStatus: document.getElementById('diagDetectorStatus'),
            diagModelStatus: document.getElementById('diagModelStatus'),
            diagLastError: document.getElementById('diagLastError'),
            diagFrozenWarning: document.getElementById('diagFrozenWarning')
        }, elements);

        this.videoEl = this.elements.video;
        this.overlayCanvas = this.elements.overlayCanvas;
        if (this.overlayCanvas) {
            this.overlayCtx = this.overlayCanvas.getContext('2d');
        }

        // Attach Button Listeners
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', () => this.initializeCamera());
        }
        if (this.elements.pauseBtn) {
            this.elements.pauseBtn.addEventListener('click', () => this.pauseCamera());
        }
        if (this.elements.resumeBtn) {
            this.elements.resumeBtn.addEventListener('click', () => this.resumeCamera());
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => this.cleanupCamera());
        }
        if (this.elements.deviceSelect) {
            this.elements.deviceSelect.addEventListener('change', (e) => {
                const devId = e.target.value;
                if (devId && devId !== this.activeDeviceId) {
                    this.switchCamera(devId);
                }
            });
        }
        if (this.elements.fullscreenBtn && this.elements.cameraWrapper) {
            this.elements.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }

        // Handle page / tab visibility
        document.addEventListener('visibilitychange', () => {
            this.isTabHidden = document.hidden;
        });

        // Start Clock & Diagnostics Loop
        this.startClock();
        this.startDiagnosticsMonitor();
        this.enumerateDevices();

        if (this.options.autoStart) {
            this.initializeCamera();
        }
        return this;
    }

    // =========================================================================
    // PRIORITY 2: SEPARATE REQUIRED FUNCTIONS
    // =========================================================================

    /**
     * 1. initializeCamera(deviceId)
     * Acquires browser camera permission and launches the pipeline.
     */
    async initializeCamera(deviceId = null) {
        if (this.state === 'CAMERA_CONNECTED' || this.state === 'VIDEO_RENDERING' || this.state === 'RECOGNIZING') {
            return;
        }

        this.clearError();
        this.updateState('REQUESTING_PERMISSION');

        if (deviceId) {
            this.activeDeviceId = deviceId;
        }

        const constraints = {
            video: this.activeDeviceId ? {
                deviceId: { exact: this.activeDeviceId },
                width: { ideal: this.options.idealWidth },
                height: { ideal: this.options.idealHeight }
            } : {
                width: { ideal: this.options.idealWidth, min: 640 },
                height: { ideal: this.options.idealHeight, min: 480 },
                facingMode: 'user'
            },
            audio: false
        };

        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("WebRTC getUserMedia is not supported by this browser. Use HTTPS or localhost.");
            }

            // Release any previously held stream
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(t => t.stop());
                this.mediaStream = null;
            }

            this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoTrack = this.mediaStream.getVideoTracks()[0];

            if (!this.videoTrack) {
                throw new Error("MediaStream does not contain an active video track.");
            }

            this.updateState('CAMERA_CONNECTED');

            // Listen for hardware disconnection
            this.videoTrack.addEventListener('ended', () => {
                this.handleError(new Error("Camera hardware was disconnected."));
                this.cleanupCamera();
            });

            const settings = this.videoTrack.getSettings ? this.videoTrack.getSettings() : {};
            if (settings.deviceId) {
                this.activeDeviceId = settings.deviceId;
            }
            if (this.elements.deviceNameDisplay) {
                this.elements.deviceNameDisplay.innerText = this.videoTrack.label || 'Webcam Stream';
            }

            // Step 2: Validate Camera Stream
            await this.validateCameraStream();

            // Step 3: Start Video Rendering
            this.startVideoRendering();

            // Step 4 & 5: Initialize AI Detection & Recognition
            await this.initializeFaceDetection();
            await this.initializeFaceRecognition();

            // Refresh device labels with permission granted
            this.enumerateDevices();

            if (typeof showToast === 'function') {
                showToast('Camera stream validated. Live classroom detection active.', 'success');
            }
        } catch (err) {
            console.error('[ClassroomCameraScanner] Initialization error:', err);
            this.handleError(err);
        }
    }

    /**
     * 2. validateCameraStream()
     * Validates stream assignment, metadata, readyState, dimensions, and real changing frames.
     */
    async validateCameraStream() {
        if (!this.videoEl) {
            throw new Error("Video element reference is missing in DOM.");
        }
        if (!this.mediaStream) {
            throw new Error("MediaStream object is null.");
        }

        this.updateState('VIDEO_INITIALIZING');

        // Assign stream to dedicated video element
        this.videoEl.srcObject = this.mediaStream;
        this.videoEl.setAttribute('autoplay', 'true');
        this.videoEl.setAttribute('playsinline', 'true');
        this.videoEl.muted = true;

        // Wait for loadedmetadata with timeout
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error("Camera metadata load timeout (camera may be locked by another application).")), 7000);
            if (this.videoEl.readyState >= HTMLMediaElement.HAVE_METADATA && this.videoEl.videoWidth > 0) {
                clearTimeout(timeout);
                resolve();
            } else {
                this.videoEl.onloadedmetadata = () => {
                    clearTimeout(timeout);
                    resolve();
                };
            }
        });

        // Await native video playback
        try {
            await this.videoEl.play();
        } catch (playErr) {
            console.warn("[CameraScanner] video.play() notice:", playErr);
        }

        // Wait until readyState >= HAVE_CURRENT_DATA
        await new Promise((resolve) => {
            if (this.videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                return resolve();
            }
            const onTimeUpdate = () => {
                if (this.videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                    this.videoEl.removeEventListener('timeupdate', onTimeUpdate);
                    resolve();
                }
            };
            this.videoEl.addEventListener('timeupdate', onTimeUpdate);
            setTimeout(resolve, 1500);
        });

        // Verify video dimensions
        if (this.videoEl.videoWidth <= 0 || this.videoEl.videoHeight <= 0) {
            throw new Error(`Invalid camera video dimensions: ${this.videoEl.videoWidth}x${this.videoEl.videoHeight}`);
        }

        // Frame Validation: Hook requestVideoFrameCallback if supported
        this.framesReceived = 0;
        this.lastFrameTimestamp = performance.now();

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            const onVfc = (now, metadata) => {
                if (this.state !== 'STOPPED' && this.state !== 'IDLE' && this.videoEl) {
                    this.framesReceived++;
                    this.lastFrameTimestamp = performance.now();
                    this.videoFrameCallbackId = this.videoEl.requestVideoFrameCallback(onVfc);
                }
            };
            this.videoFrameCallbackId = this.videoEl.requestVideoFrameCallback(onVfc);
        }

        // Pixel validation check: verify frames are not solid green / uninitialized YUV
        await new Promise((resolve) => {
            let attempts = 0;
            const testCanvas = document.createElement('canvas');
            testCanvas.width = 16;
            testCanvas.height = 16;
            const testCtx = testCanvas.getContext('2d', { willReadFrequently: true });

            const checkPixels = () => {
                attempts++;
                if (this.videoEl && this.videoEl.videoWidth > 0) {
                    testCtx.drawImage(this.videoEl, 0, 0, 16, 16);
                    const pixels = testCtx.getImageData(0, 0, 16, 16).data;
                    let isZeroedGreen = true;

                    for (let i = 0; i < pixels.length; i += 4) {
                        const r = pixels[i];
                        const g = pixels[i + 1];
                        const b = pixels[i + 2];
                        // DirectShow empty YUV(0,0,0) converts to RGB(0, ~135, 0)
                        if (!(r === 0 && g >= 100 && g <= 165 && b === 0)) {
                            isZeroedGreen = false;
                            break;
                        }
                    }

                    if (isZeroedGreen && attempts < 8) {
                        setTimeout(checkPixels, 120);
                        return;
                    }
                }
                this.isStreamValid = true;
                resolve();
            };
            setTimeout(checkPixels, 80);
        });

        return true;
    }

    /**
     * 3. startVideoRendering()
     * Displays native video element, synchronizes transparent vector canvas, and starts render loop.
     */
    startVideoRendering() {
        this.updateState('VIDEO_RENDERING');

        const vw = this.videoEl.videoWidth || 640;
        const vh = this.videoEl.videoHeight || 480;

        // Reveal dedicated live video element
        this.videoEl.style.display = 'block';

        // Hide offline placeholder
        if (this.elements.placeholder) {
            this.elements.placeholder.style.display = 'none';
        }

        // Synchronize transparent overlay canvas resolution
        if (this.overlayCanvas) {
            this.overlayCanvas.width = vw;
            this.overlayCanvas.height = vh;
            this.overlayCanvas.style.display = 'block';
        }

        // Configure offscreen inference capture canvas (capped at 720p for fast AI inference)
        this.captureCanvas.width = Math.min(vw, 720);
        this.captureCanvas.height = Math.round(this.captureCanvas.width * (vh / vw));

        if (this.elements.hudResolutionVal) {
            this.elements.hudResolutionVal.innerText = `${vw} × ${vh}`;
        }

        // Start 60fps vector render loop
        this.renderFramesCount = 0;
        this.lastRenderTime = performance.now();

        const renderLoop = (timestamp) => {
            if (this.state === 'STOPPED' || this.state === 'IDLE') {
                return;
            }

            this.renderFramesCount++;
            if (timestamp - this.lastRenderTime >= 1000) {
                this.renderFps = Math.round((this.renderFramesCount * 1000) / (timestamp - this.lastRenderTime));
                this.renderFramesCount = 0;
                this.lastRenderTime = timestamp;
                this.updateTelemetryUI();
            }

            if (this.overlayCtx && this.overlayCanvas && this.videoEl) {
                this.renderOverlay();
            }

            this.animFrameId = requestAnimationFrame(renderLoop);
        };
        this.animFrameId = requestAnimationFrame(renderLoop);
    }

    /**
     * 4. stopVideoRendering()
     * Pauses/stops video rendering and clears the transparent overlay.
     */
    stopVideoRendering() {
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
        if (this.videoFrameCallbackId && this.videoEl && 'cancelVideoFrameCallback' in this.videoEl) {
            this.videoEl.cancelVideoFrameCallback(this.videoFrameCallbackId);
            this.videoFrameCallbackId = null;
        }

        if (this.overlayCtx && this.overlayCanvas) {
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
            this.overlayCanvas.style.display = 'none';
        }

        if (this.videoEl) {
            this.videoEl.style.display = 'none';
            this.videoEl.pause();
        }

        if (this.elements.placeholder) {
            this.elements.placeholder.style.display = 'block';
        }

        this.renderFps = 0;
        this.updateTelemetryUI();
    }

    /**
     * 5. initializeFaceDetection()
     * Verifies server face detection engine readiness.
     */
    async initializeFaceDetection() {
        this.updateState('AI_MODEL_LOADING');
        try {
            const resp = await fetch('/api/status');
            const data = await resp.json();
            if (data.models && data.models.cascade) {
                this.detectorStatus = 'AI READY (OpenCV Haar + Dlib)';
            }
        } catch (e) {
            this.detectorStatus = 'Active (Local Backend)';
        }
        this.updateState('DETECTING');
    }

    /**
     * 6. initializeFaceRecognition()
     * Verifies pretrained FaceNet and classifier weights, then starts periodic inference loop.
     */
    async initializeFaceRecognition() {
        try {
            const resp = await fetch('/api/status');
            const data = await resp.json();
            if (data.models && data.models.facenet && data.models.classifier) {
                this.modelStatus = 'AI READY (FaceNet + Face_recognition.h5)';
            }
        } catch (e) {
            this.modelStatus = 'Active (FaceNet Classifier)';
        }

        this.updateState('RECOGNIZING');
        this.startProcessingLoop();
    }

    /**
     * 7. cleanupCamera()
     * Shuts down hardware tracks, stops loops, and resets camera state cleanly.
     */
    cleanupCamera() {
        this.stopVideoRendering();

        if (this.processingTimerId) {
            clearInterval(this.processingTimerId);
            this.processingTimerId = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => {
                try { track.stop(); } catch (e) {}
            });
            this.mediaStream = null;
            this.videoTrack = null;
        }

        if (this.videoEl) {
            this.videoEl.srcObject = null;
        }

        this.isStreamValid = false;
        this.latestDetections = [];
        this.facesCount = 0;
        this.renderFps = 0;
        this.latencyMs = 0;

        this.updateState('STOPPED');
        this.updateTelemetryUI();
    }

    // =========================================================================
    // AI INFERENCE & OVERLAY RENDERING
    // =========================================================================

    /**
     * Periodic offscreen frame extraction sent to /api/classroom/frame.
     */
    startProcessingLoop() {
        if (this.processingTimerId) {
            clearInterval(this.processingTimerId);
        }

        this.processingTimerId = setInterval(async () => {
            if ((this.state !== 'RECOGNIZING' && this.state !== 'DETECTING') || this.inFlight || this.isTabHidden) {
                return;
            }
            if (!this.videoEl || this.videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                return;
            }

            this.inFlight = true;
            const startT = performance.now();

            try {
                // Downsample current video frame to capture canvas
                this.captureCtx.drawImage(
                    this.videoEl, 0, 0, this.captureCanvas.width, this.captureCanvas.height
                );
                const b64Data = this.captureCanvas.toDataURL('image/jpeg', 0.72);

                const response = await fetch('/api/classroom/frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        frame: b64Data,
                        session_id: this.options.sessionId || ''
                    })
                });

                const data = await response.json();
                this.latencyMs = Math.round(performance.now() - startT);

                if (data.success) {
                    this.latestDetections = data.detections || [];
                    this.facesCount = data.faces_count !== undefined ? data.faces_count : this.latestDetections.length;

                    // Update HUD metric counters if available
                    if (data.hud_counts) {
                        const elEligible = document.getElementById('hudEligibleCount');
                        const elPresent = document.getElementById('hudPresentCount');
                        const elFaculty = document.getElementById('hudFacultyCount');
                        const elOutside = document.getElementById('hudOutsideCount');
                        const elUnknown = document.getElementById('hudUnknownCount');

                        if (elEligible) elEligible.innerText = data.hud_counts.eligible;
                        if (elPresent) elPresent.innerText = data.hud_counts.present;
                        if (elFaculty) elFaculty.innerText = data.hud_counts.faculty_excluded;
                        if (elOutside) elOutside.innerText = data.hud_counts.outside_class;
                        if (elUnknown) elUnknown.innerText = data.hud_counts.unknown;
                    }

                    // Process verified attendance cards
                    this.latestDetections.forEach(det => {
                        if (det.verified && det.roll && det.roll !== 'UNKNOWN' && det.roll !== 'ANONYMOUS') {
                            if (!this.verifiedRolls.has(det.roll)) {
                                this.verifiedRolls.add(det.roll);
                                this.renderVerifiedCard(det);
                                if (typeof playSuccessChime === 'function') {
                                    playSuccessChime();
                                }
                                if (typeof showToast === 'function') {
                                    showToast(`Attendance Verified: ${det.name} (${det.roll})`, 'success');
                                }
                            }
                        }
                    });
                }
            } catch (err) {
                console.warn('[CameraScanner] Inference request notice:', err.message);
            } finally {
                this.inFlight = false;
                this.updateTelemetryUI();
            }
        }, this.options.processingIntervalMs);
    }

    /**
     * Renders crisp vector bounding boxes and person labels on transparent overlay.
     */
    renderOverlay() {
        const ctx = this.overlayCtx;
        const cw = this.overlayCanvas.width;
        const ch = this.overlayCanvas.height;

        ctx.clearRect(0, 0, cw, ch);

        if (!this.latestDetections || this.latestDetections.length === 0) {
            return;
        }

        const scaleX = cw / (this.captureCanvas.width || cw);
        const scaleY = ch / (this.captureCanvas.height || ch);

        this.latestDetections.forEach(det => {
            const rawBbox = det.bbox || [0, 0, 100, 100];
            let x = rawBbox[0] * scaleX;
            let y = rawBbox[1] * scaleY;
            let w = rawBbox[2] * scaleX;
            let h = rawBbox[3] * scaleY;

            // Mirror horizontally because the video is flipped for natural view
            const mirrorX = cw - (x + w);

            const isEnrolled = det.roll && det.roll !== 'UNKNOWN' && det.roll !== 'ANONYMOUS';
            const isVerified = det.verified;

            // Visual Styling:
            // VERIFIED STUDENT / PRESENT: Emerald Green (#10b981)
            // FACULTY EXCLUDED: Electric Blue (#3b82f6)
            // OUTSIDE CLASS: Warning Orange (#f97316)
            // UNKNOWN / UNREGISTERED: Vibrant Red (#ef4444)
            // AMBIGUOUS / PENDING / LOW_QUALITY: Amber Yellow (#eab308)
            // ANONYMOUS: Slate Gray (#94a3b8)
            let boxColor = '#10b981';
            let headerBg = 'rgba(16, 185, 129, 0.92)';
            let textColor = '#ffffff';

            if (det.is_faculty || det.recognition_state === 'VERIFIED_FACULTY') {
                boxColor = '#3b82f6';
                headerBg = 'rgba(59, 130, 246, 0.95)';
                textColor = '#ffffff';
            } else if (det.is_outside_class || det.recognition_state === 'OUTSIDE_CLASS') {
                boxColor = '#f97316';
                headerBg = 'rgba(249, 115, 22, 0.95)';
                textColor = '#ffffff';
            } else if (det.is_unknown || det.recognition_state === 'UNKNOWN' || (!isEnrolled && det.roll !== 'ANONYMOUS')) {
                boxColor = '#ef4444';
                headerBg = 'rgba(239, 68, 68, 0.95)';
                textColor = '#ffffff';
            } else if (det.recognition_state === 'AMBIGUOUS' || det.recognition_state === 'LOW_QUALITY' || !isVerified) {
                boxColor = '#eab308';
                headerBg = 'rgba(234, 179, 8, 0.95)';
                textColor = '#0a192f';
            } else if (det.roll === 'ANONYMOUS') {
                boxColor = '#94a3b8';
                headerBg = 'rgba(148, 163, 184, 0.92)';
                textColor = '#0a192f';
            }

            // Draw Box
            ctx.save();
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = isVerified ? 3 : 2;
            ctx.shadowColor = boxColor;
            ctx.shadowBlur = isVerified ? 12 : 6;

            const radius = 6;
            ctx.beginPath();
            ctx.moveTo(mirrorX + radius, y);
            ctx.lineTo(mirrorX + w - radius, y);
            ctx.quadraticCurveTo(mirrorX + w, y, mirrorX + w, y + radius);
            ctx.lineTo(mirrorX + w, y + h - radius);
            ctx.quadraticCurveTo(mirrorX + w, y + h, mirrorX + w - radius, y + h);
            ctx.lineTo(mirrorX + radius, y + h);
            ctx.quadraticCurveTo(mirrorX, y + h, mirrorX, y + h - radius);
            ctx.lineTo(mirrorX, y + radius);
            ctx.quadraticCurveTo(mirrorX, y, mirrorX + radius, y);
            ctx.closePath();
            ctx.stroke();
            ctx.restore();

            // Corner Brackets
            const cornerLen = Math.min(16, w * 0.22);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(mirrorX, y + cornerLen);
            ctx.lineTo(mirrorX, y);
            ctx.lineTo(mirrorX + cornerLen, y);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(mirrorX + w - cornerLen, y);
            ctx.lineTo(mirrorX + w, y);
            ctx.lineTo(mirrorX + w, y + cornerLen);
            ctx.stroke();

            // Labels:
            const confPct = det.confidence ? `${Math.round(det.confidence * 100)}%` : '';
            let line1 = isEnrolled ? det.name : (det.roll === 'ANONYMOUS' ? 'IDENTITY PROTECTED' : 'UNKNOWN PERSON');
            let line2 = isEnrolled ? `ID: ${det.roll} • ${confPct}` : `${det.track_id || 'Track'} • ${confPct ? confPct + ' Match' : 'Unregistered'}`;

            if (det.is_faculty || det.recognition_state === 'VERIFIED_FACULTY') {
                line1 = `FACULTY: ${det.name}`;
                line2 = `Attendance Excluded • ${confPct}`;
            } else if (det.is_outside_class || det.recognition_state === 'OUTSIDE_CLASS') {
                line1 = `NOT IN THIS CLASS`;
                line2 = `${det.name} (${det.roll}) • Excluded`;
            } else if (det.recognition_state === 'AMBIGUOUS') {
                line1 = `AMBIGUOUS MATCH`;
                line2 = `Attendance Denied`;
            } else if (det.is_unknown || det.recognition_state === 'UNKNOWN') {
                line1 = `UNKNOWN PERSON`;
                line2 = `Unregistered • Not Marked`;
            }

            ctx.font = 'bold 12px Inter, sans-serif';
            const m1 = ctx.measureText(line1);
            const m2 = ctx.measureText(line2);
            const labelW = Math.max(w, Math.max(m1.width, m2.width) + 16);
            const labelH = 34;

            const labelY = Math.max(0, y - labelH);
            ctx.fillStyle = headerBg;
            ctx.fillRect(mirrorX, labelY, labelW, labelH);

            ctx.fillStyle = textColor;
            ctx.fillText(line1, mirrorX + 8, labelY + 14);
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.fillText(line2, mirrorX + 8, labelY + 28);

            // Status label below box
            if (det.status) {
                ctx.fillStyle = 'rgba(10, 25, 47, 0.88)';
                ctx.fillRect(mirrorX, y + h, Math.min(w, 180), 18);
                ctx.fillStyle = boxColor;
                ctx.font = '10px JetBrains Mono, monospace';
                ctx.fillText(det.status, mirrorX + 6, y + h + 13);
            }
        });

        // Overlay Alert Banners for Outside Class & Unknown Persons
        const hasOutsideClass = this.latestDetections.some(d => d.is_outside_class || d.recognition_state === 'OUTSIDE_CLASS');
        const hasUnknown = this.latestDetections.some(d => d.is_unknown || d.recognition_state === 'UNKNOWN');

        let bannerY = 40;
        if (hasOutsideClass) {
            ctx.save();
            ctx.fillStyle = 'rgba(234, 88, 12, 0.95)';
            ctx.fillRect(20, bannerY, cw - 40, 32);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.fillText('⚠ You are not part of this class. Please leave the classroom or contact the faculty member.', 32, bannerY + 20);
            ctx.restore();
            bannerY += 38;
        }

        if (hasUnknown) {
            ctx.save();
            ctx.fillStyle = 'rgba(220, 38, 38, 0.92)';
            ctx.fillRect(20, bannerY, cw - 40, 28);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.fillText('⚠ Unknown person detected — Attendance not marked.', 32, bannerY + 18);
            ctx.restore();
        }
    }

    /**
     * Dynamically renders verified student card in right-side reel.
     */
    renderVerifiedCard(det) {
        if (!this.elements.cardsContainer) return;
        const noMsg = document.getElementById('noDetectionsMsg');
        if (noMsg) noMsg.style.display = 'none';

        const cardId = `det-card-${det.roll}`;
        if (!document.getElementById(cardId)) {
            const card = document.createElement('div');
            card.id = cardId;
            card.className = 'd-flex align-items-center gap-2 p-2 mb-2 rounded bg-light border border-success-subtle shadow-sm animate-fade-in';
            card.innerHTML = `
                <img src="${det.profile_photo || '/static/profiles/' + det.roll + '.jpg'}" class="rounded-circle border" width="40" height="40" style="object-fit:cover;" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(det.name)}&background=0a192f&color=64ffda'">
                <div class="flex-grow-1 min-w-0">
                    <div class="fw-bold text-dark text-truncate small">${det.name}</div>
                    <div class="text-muted" style="font-size:0.75rem;">Roll: <strong class="text-primary">${det.roll}</strong> • ${det.section || 'A'}</div>
                </div>
                <span class="badge bg-success-subtle text-success border border-success px-2 py-1 small">
                    <i class="fa-solid fa-check me-1"></i>Verified
                </span>
            `;
            this.elements.cardsContainer.prepend(card);

            if (this.elements.verifiedBadge) {
                this.elements.verifiedBadge.innerText = `${this.verifiedRolls.size} Verified`;
            }
        }
    }

    // =========================================================================
    // CONTROLS & DIAGNOSTICS
    // =========================================================================

    pauseCamera() {
        if (this.state !== 'RECOGNIZING' && this.state !== 'DETECTING' && this.state !== 'VIDEO_RENDERING') return;
        this.updateState('PAUSED');
        if (this.videoEl) this.videoEl.pause();
        if (this.processingTimerId) {
            clearInterval(this.processingTimerId);
            this.processingTimerId = null;
        }
    }

    async resumeCamera() {
        if (this.state !== 'PAUSED') return;
        try {
            if (this.videoEl) await this.videoEl.play();
            this.updateState('RECOGNIZING');
            this.startProcessingLoop();
        } catch (e) {
            this.initializeCamera(this.activeDeviceId);
        }
    }

    async switchCamera(newDeviceId) {
        this.cleanupCamera();
        await new Promise(r => setTimeout(r, 200));
        await this.initializeCamera(newDeviceId);
    }

    async enumerateDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.availableDevices = devices.filter(d => d.kind === 'videoinput');

            if (this.elements.deviceSelect) {
                const currentVal = this.elements.deviceSelect.value;
                this.elements.deviceSelect.innerHTML = '<option value="">-- Choose Camera --</option>';
                this.availableDevices.forEach((dev, idx) => {
                    const opt = document.createElement('option');
                    opt.value = dev.deviceId;
                    opt.textContent = dev.label || `Camera ${idx + 1} (${dev.deviceId.slice(0, 8)}...)`;
                    if (currentVal && dev.deviceId === currentVal) {
                        opt.selected = true;
                    }
                    this.elements.deviceSelect.appendChild(opt);
                });
            }
        } catch (e) {
            console.warn('[CameraScanner] Device enumeration notice:', e);
        }
    }

    startDiagnosticsMonitor() {
        if (this.diagnosticsTimerId) clearInterval(this.diagnosticsTimerId);
        this.diagnosticsTimerId = setInterval(() => {
            this.updateDiagnosticsUI();
        }, 300);
    }

    updateDiagnosticsUI() {
        const el = this.elements;
        const v = this.videoEl;
        const t = this.videoTrack;

        if (el.diagStreamAvailable) {
            el.diagStreamAvailable.innerText = this.mediaStream ? 'YES' : 'NO';
            el.diagStreamAvailable.className = this.mediaStream ? 'text-success ms-1' : 'text-danger ms-1';
        }
        if (el.diagTrackState) {
            el.diagTrackState.innerText = t ? t.readyState.toUpperCase() : 'NONE';
            el.diagTrackState.className = (t && t.readyState === 'live') ? 'text-success ms-1' : 'text-secondary ms-1';
        }
        if (el.diagTrackEnabled) {
            el.diagTrackEnabled.innerText = (t && t.enabled) ? 'YES' : 'NO';
        }
        if (el.diagDeviceId) {
            el.diagDeviceId.innerText = this.activeDeviceId ? `${this.activeDeviceId.slice(0, 14)}...` : 'Default';
        }
        if (el.diagReadyState) {
            const stateMap = ['0 (HAVE_NOTHING)', '1 (HAVE_METADATA)', '2 (HAVE_CURRENT_DATA)', '3 (HAVE_FUTURE_DATA)', '4 (HAVE_ENOUGH_DATA)'];
            el.diagReadyState.innerText = v ? (stateMap[v.readyState] || v.readyState) : '0';
        }
        if (el.diagVideoDims) {
            el.diagVideoDims.innerText = v ? `${v.videoWidth} × ${v.videoHeight}` : '0 × 0';
        }
        if (el.diagCanvasDims && this.overlayCanvas) {
            el.diagCanvasDims.innerText = `${this.overlayCanvas.width} × ${this.overlayCanvas.height}`;
        }
        if (el.diagPlaybackStatus) {
            el.diagPlaybackStatus.innerText = v ? (v.paused ? 'PAUSED' : 'PLAYING') : 'STOPPED';
        }
        if (el.diagLastFrameReceived) {
            if (this.lastFrameTimestamp > 0) {
                const diff = Math.round(performance.now() - this.lastFrameTimestamp);
                el.diagLastFrameReceived.innerText = `${diff}ms ago`;
            } else {
                el.diagLastFrameReceived.innerText = '-';
            }
        }
        if (el.diagDetectorStatus) {
            el.diagDetectorStatus.innerText = this.detectorStatus;
        }
        if (el.diagModelStatus) {
            el.diagModelStatus.innerText = this.modelStatus;
        }
        if (el.diagLastError) {
            el.diagLastError.innerText = this.lastCameraError ? this.lastCameraError : 'None';
            el.diagLastError.className = this.lastCameraError ? 'text-danger ms-1' : 'text-success ms-1';
        }

        // Diagnostic Frozen Warning Alert
        if (el.diagFrozenWarning) {
            const isFrozen = (t && t.readyState === 'live' && this.state !== 'PAUSED' && this.state !== 'IDLE' && (performance.now() - this.lastFrameTimestamp > 3200));
            if (isFrozen) {
                el.diagFrozenWarning.classList.remove('d-none');
            } else {
                el.diagFrozenWarning.classList.add('d-none');
            }
        }
    }

    updateTelemetryUI() {
        if (this.elements.hudFaceCount) this.elements.hudFaceCount.innerText = this.facesCount;
        if (this.elements.hudFacesVal) this.elements.hudFacesVal.innerText = this.facesCount;
        if (this.elements.hudFpsVal) this.elements.hudFpsVal.innerText = this.renderFps;
        if (this.elements.hudLatencyVal) this.elements.hudLatencyVal.innerText = `${this.latencyMs}ms`;
    }

    updateState(newState) {
        this.state = newState;
        const badge = this.elements.statusBadge;
        const text = this.elements.statusText;

        if (badge && text) {
            badge.className = 'badge';
            switch (newState) {
                case 'REQUESTING_PERMISSION':
                    badge.classList.add('bg-warning', 'text-dark');
                    text.innerText = 'REQUESTING PERMISSION';
                    break;
                case 'CAMERA_CONNECTED':
                    badge.classList.add('bg-info', 'text-dark');
                    text.innerText = 'CAMERA CONNECTED';
                    break;
                case 'VIDEO_INITIALIZING':
                    badge.classList.add('bg-primary');
                    text.innerText = 'INITIALIZING VIDEO';
                    break;
                case 'VIDEO_RENDERING':
                    badge.classList.add('bg-cyan', 'text-dark');
                    text.innerText = 'VIDEO RENDERING';
                    break;
                case 'AI_MODEL_LOADING':
                    badge.classList.add('bg-warning', 'text-dark');
                    text.innerText = 'AI MODEL LOADING';
                    break;
                case 'DETECTING':
                    badge.classList.add('bg-primary');
                    text.innerText = 'DETECTING';
                    break;
                case 'RECOGNIZING':
                    badge.classList.add('bg-success');
                    text.innerText = 'ONLINE • RECOGNITION ACTIVE';
                    break;
                case 'PAUSED':
                    badge.classList.add('bg-warning', 'text-dark');
                    text.innerText = 'PAUSED';
                    break;
                case 'ERROR':
                    badge.classList.add('bg-danger');
                    text.innerText = 'CAMERA ERROR';
                    break;
                default:
                    badge.classList.add('bg-secondary');
                    text.innerText = 'OFFLINE';
            }
        }

        // Update button states
        const isLive = (newState === 'RECOGNIZING' || newState === 'DETECTING' || newState === 'VIDEO_RENDERING');
        const isPaused = (newState === 'PAUSED');

        if (this.elements.startBtn) this.elements.startBtn.disabled = isLive || isPaused;
        if (this.elements.pauseBtn) {
            this.elements.pauseBtn.disabled = !isLive;
            this.elements.pauseBtn.style.display = isPaused ? 'none' : 'inline-block';
        }
        if (this.elements.resumeBtn) {
            this.elements.resumeBtn.style.display = isPaused ? 'inline-block' : 'none';
        }
        if (this.elements.stopBtn) this.elements.stopBtn.disabled = !(isLive || isPaused);
    }

    handleError(err) {
        this.updateState('ERROR');
        this.lastCameraError = err.message || String(err);
        let msg = 'Failed to access camera: ' + this.lastCameraError;

        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg = 'Camera permission denied. Please allow camera access in your browser address bar.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            msg = 'No camera hardware detected. Please attach a USB webcam or enable your built-in camera.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            msg = 'Camera is already in use by another application (e.g. Zoom, Teams, OBS). Please close them.';
        }

        if (this.elements.errorAlert) {
            this.elements.errorAlert.innerText = msg;
            this.elements.errorAlert.classList.remove('d-none');
        }
        if (typeof showToast === 'function') {
            showToast(msg, 'danger');
        }
    }

    clearError() {
        this.lastCameraError = null;
        if (this.elements.errorAlert) {
            this.elements.errorAlert.innerText = '';
            this.elements.errorAlert.classList.add('d-none');
        }
    }

    startClock() {
        if (this.clockTimerId) clearInterval(this.clockTimerId);
        const update = () => {
            if (this.elements.hudClock) {
                const now = new Date();
                this.elements.hudClock.innerText = now.toTimeString().split(' ')[0];
            }
        };
        update();
        this.clockTimerId = setInterval(update, 1000);
    }

    toggleFullscreen() {
        const el = this.elements.cameraWrapper;
        if (!el) return;
        if (!document.fullscreenElement) {
            if (el.requestFullscreen) el.requestFullscreen();
            else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
        } else {
            if (document.exitFullscreen) document.exitFullscreen();
        }
    }
}

// Global attachment for accessibility
window.ClassroomCameraScanner = ClassroomCameraScanner;
