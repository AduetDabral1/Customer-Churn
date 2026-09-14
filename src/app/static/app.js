/**
 * Customer Churn Intelligence - Front-end Application Controller
 * Handles distinct Form and Prediction views, Examples dropdown, and dynamic diagnostic card rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Page Views
  const viewForm = document.getElementById('viewForm');
  const viewPrediction = document.getElementById('viewPrediction');

  // Navigation & Examples Dropdown
  const btnToggleExamples = document.getElementById('btnToggleExamples');
  const examplesDropdown = document.getElementById('examplesDropdown');
  const navBrand = document.getElementById('navBrand');
  const toast = document.getElementById('toast');

  // Example Persona Items
  const exampleHighRisk = document.getElementById('exampleHighRisk');
  const exampleLoyal = document.getElementById('exampleLoyal');
  const exampleBorderline = document.getElementById('exampleBorderline');

  // Back to Form Buttons
  const btnBackToFormTop = document.getElementById('btnBackToFormTop');
  const btnBackToFormBottom = document.getElementById('btnBackToFormBottom');

  // Form inputs
  const form = document.getElementById('churnForm');
  const tenureRange = document.getElementById('tenureRange');
  const tenureInput = document.getElementById('tenure');
  const tenureHint = document.getElementById('tenureHint');
  const monthlyChargesInput = document.getElementById('MonthlyCharges');
  const totalChargesInput = document.getElementById('TotalCharges');
  const contractSelect = document.getElementById('Contract');
  const internetSelect = document.getElementById('InternetService');
  const paymentSelect = document.getElementById('PaymentMethod');
  const paperlessSelect = document.getElementById('PaperlessBilling');
  const phoneSelect = document.getElementById('PhoneService');
  const multipleLinesSelect = document.getElementById('MultipleLines');
  const genderSelect = document.getElementById('gender');
  const seniorSelect = document.getElementById('SeniorCitizen');
  const partnerSelect = document.getElementById('Partner');
  const dependentsSelect = document.getElementById('Dependents');

  // Value-Add Service Checkboxes
  const chkOnlineSecurity = document.getElementById('chkOnlineSecurity');
  const chkOnlineBackup = document.getElementById('chkOnlineBackup');
  const chkDeviceProtection = document.getElementById('chkDeviceProtection');
  const chkTechSupport = document.getElementById('chkTechSupport');
  const chkStreamingTV = document.getElementById('chkStreamingTV');
  const chkStreamingMovies = document.getElementById('chkStreamingMovies');

  // Submit Button & Spinner
  const btnSubmit = document.getElementById('btnSubmit');
  const btnSpinner = document.getElementById('btnSpinner');

  // Output Elements on Prediction Card
  const profileId = document.getElementById('profileId');
  const chipTenure = document.getElementById('chipTenure');
  const chipMonthly = document.getElementById('chipMonthly');
  const chipContract = document.getElementById('chipContract');
  const predictionBanner = document.getElementById('predictionBanner');
  const tierBadge = document.getElementById('tierBadge');
  const predictionOutcome = document.getElementById('predictionOutcome');
  const predictionDesc = document.getElementById('predictionDesc');
  const gaugeCircle = document.getElementById('gaugeCircle');
  const gaugeValue = document.getElementById('gaugeValue');
  const detailProb = document.getElementById('detailProb');
  const detailTier = document.getElementById('detailTier');
  const detailLatency = document.getElementById('detailLatency');
  const kpiContainer = document.getElementById('kpiContainer');
  const adviceList = document.getElementById('adviceList');
  const curvePoint = document.getElementById('curvePoint');
  const predictionTimestamp = document.getElementById('predictionTimestamp');

  // =========================================================================
  // 1. View Switching (Distinct Pages: Form vs. Prediction)
  // =========================================================================
  function switchPage(page) {
    if (page === 'form') {
      viewPrediction.classList.remove('active');
      viewPrediction.classList.add('hidden');
      
      viewForm.classList.remove('hidden');
      viewForm.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (page === 'prediction') {
      viewForm.classList.remove('active');
      viewForm.classList.add('hidden');
      
      viewPrediction.classList.remove('hidden');
      viewPrediction.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  // Back buttons return to form
  btnBackToFormTop.addEventListener('click', () => switchPage('form'));
  btnBackToFormBottom.addEventListener('click', () => switchPage('form'));
  navBrand.addEventListener('click', () => switchPage('form'));

  // =========================================================================
  // 2. "See Examples" Dropdown & Toast Notification
  // =========================================================================
  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2800);
  }

  btnToggleExamples.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = examplesDropdown.classList.contains('open');
    if (isOpen) {
      examplesDropdown.classList.remove('open');
      btnToggleExamples.classList.remove('active');
    } else {
      examplesDropdown.classList.add('open');
      btnToggleExamples.classList.add('active');
    }
  });

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.examples-menu-wrapper')) {
      examplesDropdown.classList.remove('open');
      btnToggleExamples.classList.remove('active');
    }
  });

  // Apply Persona Data to Form
  const checkPills = document.querySelectorAll('.checkbox-pill-label');
  checkPills.forEach(pill => {
    const cb = pill.querySelector('input[type="checkbox"]');
    cb.addEventListener('change', () => {
      pill.classList.toggle('active', cb.checked);
    });
  });

  function applyPreset(preset, name) {
    genderSelect.value = preset.gender;
    seniorSelect.value = preset.SeniorCitizen;
    partnerSelect.value = preset.Partner;
    dependentsSelect.value = preset.Dependents;
    
    updateTenure(preset.tenure);
    contractSelect.value = preset.Contract;
    paymentSelect.value = preset.PaymentMethod;
    paperlessSelect.value = preset.PaperlessBilling;

    internetSelect.value = preset.InternetService;
    phoneSelect.value = preset.PhoneService;
    multipleLinesSelect.value = preset.MultipleLines;

    chkOnlineSecurity.checked = preset.OnlineSecurity === "Yes";
    chkOnlineBackup.checked = preset.OnlineBackup === "Yes";
    chkDeviceProtection.checked = preset.DeviceProtection === "Yes";
    chkTechSupport.checked = preset.TechSupport === "Yes";
    chkStreamingTV.checked = preset.StreamingTV === "Yes";
    chkStreamingMovies.checked = preset.StreamingMovies === "Yes";

    checkPills.forEach(pill => {
      const cb = pill.querySelector('input[type="checkbox"]');
      pill.classList.toggle('active', cb.checked);
    });

    monthlyChargesInput.value = preset.MonthlyCharges.toFixed(2);
    totalChargesInput.value = preset.TotalCharges.toFixed(2);

    // Close dropdown and return to form view so user can review values
    examplesDropdown.classList.remove('open');
    btnToggleExamples.classList.remove('active');
    switchPage('form');
    showToast(`Loaded Example: ${name}`);
  }

  exampleHighRisk.addEventListener('click', () => {
    applyPreset({
      gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No",
      tenure: 2, Contract: "Month-to-month", PaymentMethod: "Electronic check",
      PaperlessBilling: "Yes", InternetService: "Fiber optic", PhoneService: "Yes",
      MultipleLines: "No", OnlineSecurity: "No", OnlineBackup: "No",
      DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
      StreamingMovies: "Yes", MonthlyCharges: 95.50, TotalCharges: 191.00
    }, "⚡ High Risk Persona");
  });

  exampleLoyal.addEventListener('click', () => {
    applyPreset({
      gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes",
      tenure: 60, Contract: "Two year", PaymentMethod: "Bank transfer (automatic)",
      PaperlessBilling: "No", InternetService: "DSL", PhoneService: "Yes",
      MultipleLines: "Yes", OnlineSecurity: "Yes", OnlineBackup: "Yes",
      DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No",
      StreamingMovies: "No", MonthlyCharges: 64.20, TotalCharges: 3852.00
    }, "🛡️ Loyal Persona");
  });

  exampleBorderline.addEventListener('click', () => {
    applyPreset({
      gender: "Female", SeniorCitizen: 1, Partner: "Yes", Dependents: "No",
      tenure: 14, Contract: "Month-to-month", PaymentMethod: "Credit card (automatic)",
      PaperlessBilling: "Yes", InternetService: "Fiber optic", PhoneService: "Yes",
      MultipleLines: "No", OnlineSecurity: "No", OnlineBackup: "Yes",
      DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
      StreamingMovies: "No", MonthlyCharges: 80.00, TotalCharges: 1120.00
    }, "⚖️ Borderline Persona");
  });

  // =========================================================================
  // 3. Synchronize Tenure Slider & Live Spend Calculation
  // =========================================================================
  function updateTenure(val) {
    const tenure = parseInt(val, 10);
    tenureRange.value = tenure;
    tenureInput.value = tenure;
    
    if (tenure <= 6) {
      tenureHint.textContent = "Critical early window (< 6 mos)";
      tenureHint.style.color = "#EF4444";
    } else if (tenure <= 24) {
      tenureHint.textContent = "Growth phase (6-24 mos)";
      tenureHint.style.color = "#F59E0B";
    } else {
      tenureHint.textContent = "Established subscriber (> 2 yrs)";
      tenureHint.style.color = "#10B981";
    }

    const monthly = parseFloat(monthlyChargesInput.value) || 0;
    const computedTotal = Math.max(monthly, Math.round(monthly * Math.max(1, tenure) * 100) / 100);
    totalChargesInput.value = computedTotal.toFixed(2);
  }

  tenureRange.addEventListener('input', (e) => updateTenure(e.target.value));
  tenureInput.addEventListener('input', (e) => updateTenure(e.target.value));

  monthlyChargesInput.addEventListener('input', () => {
    const tenure = parseInt(tenureInput.value, 10) || 1;
    const monthly = parseFloat(monthlyChargesInput.value) || 0;
    totalChargesInput.value = Math.max(monthly, Math.round(monthly * Math.max(1, tenure) * 100) / 100).toFixed(2);
  });

  // =========================================================================
  // 4. Build Payload from Form
  // =========================================================================
  function buildPayload() {
    const hasInternet = internetSelect.value !== "No";
    
    return {
      gender: genderSelect.value,
      SeniorCitizen: parseInt(seniorSelect.value, 10),
      Partner: partnerSelect.value,
      Dependents: dependentsSelect.value,
      tenure: parseInt(tenureInput.value, 10),
      PhoneService: phoneSelect.value,
      MultipleLines: phoneSelect.value === "No" ? "No phone service" : multipleLinesSelect.value,
      InternetService: internetSelect.value,
      OnlineSecurity: !hasInternet ? "No internet service" : (chkOnlineSecurity.checked ? "Yes" : "No"),
      OnlineBackup: !hasInternet ? "No internet service" : (chkOnlineBackup.checked ? "Yes" : "No"),
      DeviceProtection: !hasInternet ? "No internet service" : (chkDeviceProtection.checked ? "Yes" : "No"),
      TechSupport: !hasInternet ? "No internet service" : (chkTechSupport.checked ? "Yes" : "No"),
      StreamingTV: !hasInternet ? "No internet service" : (chkStreamingTV.checked ? "Yes" : "No"),
      StreamingMovies: !hasInternet ? "No internet service" : (chkStreamingMovies.checked ? "Yes" : "No"),
      Contract: contractSelect.value,
      PaperlessBilling: paperlessSelect.value,
      PaymentMethod: paymentSelect.value,
      MonthlyCharges: parseFloat(monthlyChargesInput.value),
      TotalCharges: parseFloat(totalChargesInput.value)
    };
  }

  // =========================================================================
  // 5. Submit Form & Transition to Prediction Page
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    btnSubmit.disabled = true;
    btnSpinner.style.display = 'inline-block';
    const startTime = performance.now();

    const payload = buildPayload();

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Inference server responded with ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const latency = Math.round(performance.now() - startTime);

      // Render all diagnostic card sections
      renderDiagnosticCard(result, payload, latency);

      // Transition to Prediction Page
      switchPage('prediction');

    } catch (err) {
      console.error("Prediction execution failed:", err);
      alert(`Inference failed: ${err.message}`);
    } finally {
      btnSubmit.disabled = false;
      btnSpinner.style.display = 'none';
    }
  });

  // =========================================================================
  // 6. Render Master Prediction Card (Image 2 Diagnostic Architecture)
  // =========================================================================
  function renderDiagnosticCard(result, payload, latency) {
    const prob = result.churn_probability;
    const threshold = result.threshold_used;
    const isAtRisk = prob >= threshold;

    // Generate randomized customer ID
    const randomId = Math.floor(1000 + Math.random() * 9000);
    profileId.textContent = `ID: #TX-${randomId}`;
    predictionTimestamp.textContent = `Analysis generated: ${new Date().toLocaleTimeString()}`;

    // Customer Profile Chips
    chipTenure.textContent = `${payload.tenure} mos`;
    chipMonthly.textContent = `$${payload.MonthlyCharges.toFixed(2)}`;
    chipContract.textContent = payload.Contract;

    // Qualitative Prediction Banner
    predictionBanner.className = 'prediction-banner ' + 
      (result.risk_tier === 'HIGH' ? 'risk-high' : (result.risk_tier === 'MEDIUM' ? 'risk-medium' : 'risk-low'));

    tierBadge.className = 'tier-badge ' + 
      (result.risk_tier === 'HIGH' ? 'tier-high' : (result.risk_tier === 'MEDIUM' ? 'tier-medium' : 'tier-low'));
    tierBadge.textContent = `${result.risk_tier} RISK`;

    if (isAtRisk) {
      predictionOutcome.textContent = "More Likely to Churn";
      predictionOutcome.style.color = result.risk_tier === 'HIGH' ? '#EF4444' : '#B45309';
      predictionDesc.textContent = `Subscriber exhibits elevated churn probability (${(prob * 100).toFixed(1)}%) exceeding the ${Math.round(threshold * 100)}% calibrated business threshold. Immediate retention intervention is recommended.`;
    } else {
      predictionOutcome.textContent = "Not Likely to Churn (Retained)";
      predictionOutcome.style.color = '#10B981';
      predictionDesc.textContent = `Subscriber has low churn risk (${(prob * 100).toFixed(1)}%), well below the ${Math.round(threshold * 100)}% threshold. Account is considered stable and healthy.`;
    }

    // Probability Radial Gauge
    const circumference = 345;
    const offset = circumference - (circumference * prob);
    gaugeCircle.style.strokeDashoffset = offset;
    gaugeCircle.style.stroke = result.risk_tier === 'HIGH' ? '#EF4444' : (result.risk_tier === 'MEDIUM' ? '#FCA311' : '#10B981');
    gaugeValue.textContent = `${(prob * 100).toFixed(1)}%`;
    gaugeValue.style.color = result.risk_tier === 'HIGH' ? '#EF4444' : '#14213D';

    detailProb.textContent = prob.toFixed(4);
    detailTier.textContent = result.risk_tier;
    detailLatency.textContent = `${latency} ms`;

    // Render KPI Cards (Contributing Factors)
    renderKPICards(payload, prob);

    // Render Follow-up Actions & Retention Playbook
    renderAdvicePlaybook(payload, isAtRisk, result.risk_tier);

    // Update Tenure Risk Projection Spline
    updateRiskCurve(payload.tenure, prob);
  }

  // Key Factors to Consider (KPI Cards)
  function renderKPICards(payload, prob) {
    kpiContainer.innerHTML = '';
    const factors = [];

    // Factor 1: Contract Exposure
    if (payload.Contract === "Month-to-month") {
      factors.push({
        accent: "accent-danger",
        category: "Contract Commitment",
        status: "High Exposure",
        title: "Month-to-month Agreement",
        desc: "Zero cancellation friction. Month-to-month customers churn at 42.7% baseline across the cohort."
      });
    } else {
      factors.push({
        accent: "accent-success",
        category: "Contract Commitment",
        status: "Strong Anchor",
        title: `${payload.Contract} Agreement`,
        desc: "Multi-month commitments reduce churn probability by over 65% relative to month-to-month."
      });
    }

    // Factor 2: Support & Security Suite
    const hasTechSupport = payload.TechSupport === "Yes";
    const hasSecurity = payload.OnlineSecurity === "Yes";
    if (!hasTechSupport && !hasSecurity && payload.InternetService !== "No") {
      factors.push({
        accent: "accent-gold",
        category: "Support & Security Shield",
        status: "Unprotected",
        title: "No Tech Support or Online Security",
        desc: "Subscribers without security add-ons churn 3.2x faster when encountering technical bottlenecks."
      });
    } else if (hasTechSupport && hasSecurity) {
      factors.push({
        accent: "accent-success",
        category: "Product Stickiness",
        status: "High Engagement",
        title: "Active Support & Security Suite",
        desc: "Customers utilizing both Tech Support and Security exhibit maximum lifetime retention."
      });
    } else {
      factors.push({
        accent: "accent-navy",
        category: "Service Utilization",
        status: "Partial Shield",
        title: "Partial Protection Coverage",
        desc: "Customer has some add-on coverage; expanding to full support bundle improves long-term loyalty."
      });
    }

    // Factor 3: Billing & Payment Friction
    if (payload.PaymentMethod === "Electronic check") {
      factors.push({
        accent: "accent-danger",
        category: "Payment Channel Friction",
        status: "Manual Friction",
        title: "Manual Electronic Check Billing",
        desc: "Electronic check payers churn at almost 2x the rate of automated bank debit/credit card subscribers."
      });
    } else if (payload.MonthlyCharges > 80 && payload.InternetService === "Fiber optic") {
      factors.push({
        accent: "accent-gold",
        category: "Spend Sensitivity",
        status: "Premium Tier",
        title: `High Tier Spend ($${payload.MonthlyCharges.toFixed(2)}/mo)`,
        desc: "High monthly fiber charges heighten price sensitivity and competitor shopping."
      });
    } else {
      factors.push({
        accent: "accent-navy",
        category: "Financial Health",
        status: "Balanced",
        title: `Stable Monthly Billing ($${payload.MonthlyCharges.toFixed(2)}/mo)`,
        desc: `Automated payment method (${payload.PaymentMethod}) minimizes recurring payment drop-off.`
      });
    }

    factors.forEach(f => {
      const card = document.createElement('div');
      card.className = `kpi-card ${f.accent}`;
      card.innerHTML = `
        <div class="kpi-meta-top">
          <span>${f.category}</span>
          <strong>${f.status}</strong>
        </div>
        <div class="kpi-title">${f.title}</div>
        <div class="kpi-subtext">${f.desc}</div>
      `;
      kpiContainer.appendChild(card);
    });
  }

  // Prescriptive Retention Playbook
  function renderAdvicePlaybook(payload, isAtRisk, riskTier) {
    adviceList.innerHTML = '';
    const steps = [];

    if (isAtRisk) {
      if (payload.Contract === "Month-to-month") {
        steps.push({
          title: "Step 1: Offer 1-Year Contract Lock-in Discount",
          desc: "Extend a $15/month promotional credit in exchange for converting to a 12-month contract commitment."
        });
      } else {
        steps.push({
          title: "Step 1: Conduct Proactive VIP Check-in",
          desc: "Schedule a high-priority CS health-check call within 24 hours to address service satisfaction."
        });
      }

      if (payload.TechSupport !== "Yes") {
        steps.push({
          title: "Step 2: Pitch Complimentary Tech Support Bundle",
          desc: "Enroll subscriber in 6 months of free 24/7 dedicated Tech Support to boost product stickiness."
        });
      } else {
        steps.push({
          title: "Step 2: Review Bandwidth & Device Performance",
          desc: "Ensure fiber connectivity meets throughput SLAs to eliminate performance-driven churn."
        });
      }

      if (payload.PaymentMethod === "Electronic check") {
        steps.push({
          title: "Step 3: Migrate to Auto-Debit with Incentive",
          desc: "Offer a one-time $10 account credit to set up automated recurring payment, cutting friction."
        });
      } else {
        steps.push({
          title: "Step 3: Monitor Usage & Engagement Telemetry",
          desc: "Track support ticket volume and login frequencies over the next 30 days."
        });
      }
    } else {
      steps.push({
        title: "Step 1: Retain & Maintain Healthy Engagement",
        desc: "Subscriber is on a secure plan. No urgent intervention required; maintain standard service quality."
      });
      steps.push({
        title: "Step 2: Explore Value-Add Cross-Sell",
        desc: "Consider introducing streaming service discounts or high-speed add-ons during next billing cycle."
      });
      steps.push({
        title: "Step 3: Request Customer Advocacy / Review",
        desc: "Invite subscriber to participate in our Net Promoter Score (NPS) loyalty review program."
      });
    }

    steps.forEach(s => {
      const item = document.createElement('div');
      item.className = 'advice-item';
      item.innerHTML = `
        <div class="advice-icon">✓</div>
        <div class="advice-content">
          <h5>${s.title}</h5>
          <p>${s.desc}</p>
        </div>
      `;
      adviceList.appendChild(item);
    });
  }

  // Update Tenure Curve Position
  function updateRiskCurve(tenure, prob) {
    const x = 10 + (Math.min(tenure, 72) / 72) * 340;
    const y = 85 - (prob * 65);
    curvePoint.setAttribute('cx', Math.round(x));
    curvePoint.setAttribute('cy', Math.round(y));
  }

  // Initialize Default Form State
  updateTenure(tenureInput.value);
});
