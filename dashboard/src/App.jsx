import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
  TrendingUp,
  Store,
  Calendar,
  Percent,
  Info,
  Activity,
  CheckCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
  Sliders,
  Sparkles
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  // States
  const [stores, setStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState(1);
  const [startDate, setStartDate] = useState('2015-06-01');
  const [duration, setDuration] = useState(30); // 7, 14, 30 days
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Model & Server Health States
  const [health, setHealth] = useState({ online: false, modelLoaded: false, datasetsLoaded: false });
  const [modelInfo, setModelInfo] = useState(null);
  
  // Forecast Data States
  const [forecastData, setForecastData] = useState(null);
  const [overrides, setOverrides] = useState([]); // Array of {date, open, promo, school_holiday, state_holiday}
  const [lastCalculatedKey, setLastCalculatedKey] = useState("");

  // Get date limits and stores list on startup
  useEffect(() => {
    checkServerHealth();
    fetchStores();
  }, []);

  // Fetch forecast when store, date, duration, or overrides change
  useEffect(() => {
    if (health.online) {
      handleGenerateForecast(false);
    }
  }, [selectedStore, startDate, duration, health.online]);

  const checkServerHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      if (res.status === 200) {
        setHealth({
          online: true,
          modelLoaded: res.data.model_loaded,
          datasetsLoaded: res.data.datasets_loaded
        });
        
        // Fetch model info
        const infoRes = await axios.get(`${API_BASE_URL}/model-info`);
        setModelInfo(infoRes.data);
      }
    } catch (err) {
      console.error("Server health check failed:", err);
      setHealth({ online: false, modelLoaded: false, datasetsLoaded: false });
      setError("Cannot connect to Python FastAPI backend. Ensure the backend server is running.");
    }
  };

  const fetchStores = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/stores`);
      setStores(res.data);
    } catch (err) {
      console.error("Failed to fetch stores list:", err);
    }
  };

  // Generate End Date based on start date and duration
  const calculatedEndDate = useMemo(() => {
    try {
      const start = new Date(startDate);
      if (isNaN(start.getTime())) return startDate;
      start.setDate(start.getDate() + duration - 1);
      return start.toISOString().split('T')[0];
    } catch (e) {
      return startDate;
    }
  }, [startDate, duration]);

  // Reset overrides when changing inputs
  const resetOverrides = () => {
    setOverrides([]);
    handleGenerateForecast(true); // force generate with empty overrides
  };

  // Main forecast generator call
  const handleGenerateForecast = async (forceEmptyOverrides = false) => {
    const activeOverrides = forceEmptyOverrides ? [] : overrides;
    
    // Create unique key to prevent redundant requests
    const requestKey = `${selectedStore}-${startDate}-${duration}-${JSON.stringify(activeOverrides)}`;
    if (requestKey === lastCalculatedKey) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        store_id: Number(selectedStore),
        start_date: startDate,
        end_date: calculatedEndDate,
        overrides: activeOverrides.length > 0 ? activeOverrides : null
      });

      setForecastData(response.data);
      setLastCalculatedKey(requestKey);

      // If overrides were empty, pre-populate the interactive overrides state with the baseline schedule
      if (activeOverrides.length === 0) {
        const baseline = response.data.predictions.map(p => ({
          date: p.date,
          open: p.open,
          promo: p.promo,
          school_holiday: p.school_holiday,
          state_holiday: p.state_holiday
        }));
        setOverrides(baseline);
      }
    } catch (err) {
      console.error("Forecast error:", err);
      const msg = err.response?.data?.detail || "An error occurred while generating the forecast.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // Modify individual schedule overrides and trigger recount
  const handleToggleOverride = (date, field) => {
    const updated = overrides.map(item => {
      if (item.date === date) {
        const currentVal = item[field];
        return {
          ...item,
          [field]: currentVal === 1 ? 0 : 1
        };
      }
      return item;
    });
    setOverrides(updated);
    
    // Proactively send updated overrides
    triggerForecastWithCustomOverrides(updated);
  };

  const triggerForecastWithCustomOverrides = async (updatedOverrides) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        store_id: Number(selectedStore),
        start_date: startDate,
        end_date: calculatedEndDate,
        overrides: updatedOverrides
      });
      setForecastData(response.data);
      setLastCalculatedKey(`${selectedStore}-${startDate}-${duration}-${JSON.stringify(updatedOverrides)}`);
    } catch (err) {
      console.error("Forecast override error:", err);
      const msg = err.response?.data?.detail || "Error recalculating overrides.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // Compute stats for display
  const dashboardStats = useMemo(() => {
    if (!forecastData) return { totalSales: 0, avgSales: 0, promoDays: 0, closedDays: 0 };
    
    const preds = forecastData.predictions;
    const totalSales = preds.reduce((acc, p) => acc + p.predicted_sales, 0);
    const openDays = preds.filter(p => p.open === 1).length;
    const avgSales = openDays > 0 ? (totalSales / openDays) : 0;
    const promoDays = preds.filter(p => p.promo === 1).length;
    const closedDays = preds.filter(p => p.open === 0).length;

    return {
      totalSales: Math.round(totalSales),
      avgSales: Math.round(avgSales),
      promoDays,
      closedDays
    };
  }, [forecastData]);

  // Format Recharts data (combine history + predictions)
  const chartData = useMemo(() => {
    if (!forecastData) return [];
    
    const list = [];
    
    // 1. Add historical sales (30 days prior)
    forecastData.historical_sales.forEach(h => {
      list.push({
        date: h.date,
        type: 'History',
        'Recent Sales': h.sales,
        'Predicted Sales': null,
        'Actual Sales': null
      });
    });
    
    // 2. Add prediction sales
    forecastData.predictions.forEach(p => {
      list.push({
        date: p.date,
        type: 'Forecast',
        'Recent Sales': null,
        'Predicted Sales': p.predicted_sales,
        'Actual Sales': p.actual_sales
      });
    });
    
    return list;
  }, [forecastData]);

  const selectedStoreDetails = useMemo(() => {
    return stores.find(s => s.store_id === Number(selectedStore)) || null;
  }, [stores, selectedStore]);

  return (
    <div className="app-container">
      {/* Header bar with gradient glow line beneath */}
      <header className="app-header">
        <div className="app-title">
          <TrendingUp size={20} />
          <span>
            <span className="title-logo-gradient">ROSSMANN</span> SALES INTELLIGENCE
          </span>
        </div>
        
        {/* API Status indicator rounded pill */}
        <div className="api-status-pill">
          <Activity size={12} className="text-gray-400" />
          <span className={`status-dot ${health.online ? 'active' : ''}`}></span>
          <span>{health.online ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}</span>
        </div>
      </header>

      {/* Main Content Shell */}
      <main className="app-content">
        
        {/* Hero Section (Introduction Area) */}
        <section className="hero-section">
          <div className="hero-content">
            <span className="hero-tagline">Predictive Retail Modelling</span>
            <h1 className="hero-title">Demand Forecasting & Sales Intelligence</h1>
            <p className="hero-description">
              Leverage recursive machine learning pipelines to forecast store-level sales volumes. Toggle promotional runs, regional calendars, and operating states to test scenario variations.
            </p>
          </div>
          
          {/* Faint financial line-art vector decoration */}
          <svg className="hero-graphic-vector" viewBox="0 0 200 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 80 Q 50 60, 90 75 T 170 30" stroke="url(#hero-trend-gradient)" strokeWidth="3" strokeLinecap="round" />
            <circle cx="170" cy="30" r="4" fill="#8b5cf6" />
            <circle cx="90" cy="75" r="3" fill="#6366f1" />
            <circle cx="50" cy="60" r="3" fill="#3b82f6" />
            <defs>
              <linearGradient id="hero-trend-gradient" x1="10" y1="80" x2="170" y2="30" gradientUnits="userSpaceOnUse">
                <stop stopColor="#3b82f6" />
                <stop offset="0.5" stopColor="#6366f1" />
                <stop offset="1" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
          </svg>
        </section>

        {/* Error Alert Card */}
        {error && (
          <div className="alert-card">
            <div className="alert-badge">
              <AlertTriangle size={18} />
            </div>
            <div>
              <div className="error-title font-semibold text-sm">System Connection Alert</div>
              <div className="error-desc text-xs">{error}</div>
            </div>
            <button onClick={checkServerHealth} className="btn-alert-retry">
              <RefreshCw size={11} /> Reconnect
            </button>
          </div>
        )}

        {/* Split Grid: Configuration & Metrics on top, Chart centerpiece beneath */}
        <div className="dashboard-grid">
          {/* Sidebar / Configuration card */}
          <aside className="controls-sidebar">
            <div className="parameters-card">
              <span className="section-label">
                <Sliders size={11} />
                CONFIGURATION
              </span>
              <h2 className="panel-heading">Parameters</h2>

              {/* Form Controls with STRICT vertical breathing space */}
              <div className="form-group">
                <label className="form-label">
                  <Store size={11} />
                  Store Selection
                </label>
                <div className="custom-select-wrapper">
                  <select 
                    className="form-select"
                    value={selectedStore}
                    onChange={(e) => setSelectedStore(Number(e.target.value))}
                    disabled={loading || !health.online}
                  >
                    {stores.length > 0 ? (
                      stores.map(s => (
                        <option key={s.store_id} value={s.store_id}>
                          Store {s.store_id} ({s.store_type.toUpperCase()} / {s.assortment.toUpperCase()})
                        </option>
                      ))
                    ) : (
                      <option value={1}>Store 1</option>
                    )}
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <Calendar size={11} />
                  Target Start Date
                </label>
                <input 
                  type="date"
                  className="form-input"
                  value={startDate}
                  min={modelInfo?.date_min || "2013-01-31"}
                  max={modelInfo?.date_max || "2015-07-31"}
                  onChange={(e) => setStartDate(e.target.value)}
                  disabled={loading || !health.online}
                />
                <span className="text-[10px] text-slate-500 mt-1.5 block">
                  Training limit: {modelInfo?.date_min || '2013-01-31'} to {modelInfo?.date_max || '2015-07-31'}
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <Activity size={11} />
                  Forecast Horizon
                </label>
                <div className="duration-toggle-group">
                  {[7, 14, 30].map(d => (
                    <button
                      key={d}
                      type="button"
                      className={`btn-toggle ${duration === d ? 'active' : ''}`}
                      onClick={() => setDuration(d)}
                      disabled={loading || !health.online}
                    >
                      {d} Days
                    </button>
                  ))}
                </div>
              </div>

              {/* Prominent Action CTA Button */}
              <button
                type="button"
                className="btn-cta mt-5"
                onClick={() => handleGenerateForecast(false)}
                disabled={loading || !health.online}
              >
                {loading ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
                Run Analytics Model
              </button>
            </div>

            {/* Store Information Profile Metadata */}
            <div className="store-details-card">
              <span className="section-label">
                <Store size={11} />
                METADATA
              </span>
              <h2 className="panel-heading">Store Profile</h2>
              
              {forecastData?.store_info ? (
                <div>
                  <div className="info-row">
                    <span className="info-label">Profile ID</span>
                    <span className="info-val">#{forecastData.store_info.store_id}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Format Type</span>
                    <span className="info-val">Type {forecastData.store_info.store_type.toUpperCase()}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Assortment</span>
                    <span className="info-val">Class {forecastData.store_info.assortment.toUpperCase()}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Competition</span>
                    <span className="info-val">
                      {forecastData.store_info.competition_distance 
                        ? `${forecastData.store_info.competition_distance.toLocaleString()}m`
                        : 'None'}
                    </span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Promo2 enroll</span>
                    <span className="info-val">
                      {forecastData.store_info.promo2 === 1 ? (
                        <span className="badge badge-success">Enrolled</span>
                      ) : (
                        <span className="badge badge-danger">Inactive</span>
                      )}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-xs text-slate-500">
                  Execute prediction to load store profile
                </div>
              )}
            </div>
          </aside>

          {/* Right Column: Business Metrics & Centerpiece Forecast Chart */}
          <section className="analytics-area">
            {/* KPI Cards Row */}
            <div className="kpi-row">
              {/* Card 1: Forecast Total */}
              <div className="kpi-card indigo">
                <div className="kpi-header">
                  <span className="section-label">Forecast Total</span>
                  <div className="kpi-icon-badge"><TrendingUp size={15} /></div>
                </div>
                <div>
                  <div className="kpi-number-heavy">₹ {dashboardStats.totalSales.toLocaleString()}</div>
                  <div className="kpi-subtext">Summed sales over {duration} days</div>
                </div>
              </div>

              {/* Card 2: Operating average */}
              <div className="kpi-card emerald">
                <div className="kpi-header">
                  <span className="section-label">Operating average</span>
                  <div className="kpi-icon-badge"><Activity size={15} /></div>
                </div>
                <div>
                  <div className="kpi-number-heavy">₹ {dashboardStats.avgSales.toLocaleString()}</div>
                  <div className="kpi-subtext">Averaged over operating open days</div>
                </div>
              </div>

              {/* Card 3: Backtest accuracy */}
              <div className="kpi-card violet">
                <div className="kpi-header">
                  <span className="section-label">Backtest accuracy</span>
                  <div className="kpi-icon-badge"><Percent size={15} /></div>
                </div>
                <div>
                  <div className="kpi-number-heavy">
                    {forecastData?.metrics?.wape ? `${(100 - forecastData.metrics.wape).toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="kpi-subtext">
                    {forecastData?.metrics ? (
                      <span className="text-emerald-400 font-semibold flex items-center gap-0.5">
                        <CheckCircle size={10} /> R²: {forecastData.metrics.r2}
                      </span>
                    ) : (
                      'No validation history'
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Forecast Chart Panel (Main Visual Piece) */}
            <div className="chart-hero-card">
              <div className="chart-header">
                <div>
                  <span className="section-label">
                    <Activity size={11} />
                    DEMAND FORECAST
                  </span>
                  <h3 className="panel-heading" style={{ marginBottom: 0 }}>Demand Forecasting & Historical Comparison</h3>
                </div>
                
                {/* Custom Pill Legends */}
                <div className="legend-pill-group">
                  <div className="legend-pill history">
                    <span className="legend-dot-indicator"></span>
                    <span>Recent History</span>
                  </div>
                  <div className="legend-pill forecast">
                    <span className="legend-dot-indicator"></span>
                    <span>Model Forecast</span>
                  </div>
                  {forecastData?.predictions[0]?.actual_sales !== null && (
                    <div className="legend-pill actual">
                      <span className="legend-dot-indicator"></span>
                      <span>Actual Sales</span>
                    </div>
                  )}
                </div>
              </div>

              {loading ? (
                <div className="loading-overlay" style={{ height: '400px' }}>
                  <div className="spinner"></div>
                  <p className="font-semibold text-xs mt-2 text-indigo-400">Generating scenario forecast...</p>
                </div>
              ) : chartData.length > 0 ? (
                <div className="chart-container-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -5, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.03)" vertical={false} />
                      <XAxis 
                        dataKey="date" 
                        stroke="#64748b" 
                        fontSize={10} 
                        tickLine={false}
                        axisLine={false}
                        dy={8}
                      />
                      <YAxis 
                        stroke="#64748b" 
                        fontSize={10}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(v) => `₹${v}`}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          background: '#0f1626', 
                          border: '1px solid rgba(99, 102, 241, 0.2)',
                          borderRadius: '8px',
                          color: '#f8fafc',
                          fontSize: '12px',
                          boxShadow: '0 10px 25px rgba(0,0,0,0.5)'
                        }} 
                        labelStyle={{ fontWeight: 'bold', color: '#94a3b8' }}
                      />
                      <ReferenceLine 
                        x={startDate} 
                        stroke="#ef4444" 
                        strokeDasharray="3 3" 
                        strokeWidth={1} 
                        label={{ value: 'Forecast Start', fill: '#ef4444', fontSize: 10, position: 'insideTopLeft', dy: 10 }} 
                      />
                      
                      {/* Historical Sales Series */}
                      <Line 
                        name="Recent Sales"
                        type="monotone" 
                        dataKey="Recent Sales" 
                        stroke="#475569" 
                        strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                      
                      {/* Forecast Sales Series */}
                      <Line 
                        name="Predicted Sales"
                        type="monotone" 
                        dataKey="Predicted Sales" 
                        stroke="#6366f1" 
                        strokeWidth={3}
                        dot={duration <= 14}
                        activeDot={{ r: 5 }}
                      />
                      
                      {/* Actual Sales Series */}
                      <Line 
                        name="Actual Sales"
                        type="monotone" 
                        dataKey="Actual Sales" 
                        stroke="#10b981" 
                        strokeWidth={2.5}
                        strokeDasharray="4 4"
                        dot={duration <= 14}
                        activeDot={{ r: 4 }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                /* Designed empty visual state */
                <div className="empty-state-illustration-wrapper" style={{ height: '400px' }}>
                  <div className="empty-state-visual-circle">
                    <TrendingUp />
                  </div>
                  <div className="empty-state-title">No Forecast Data Loaded</div>
                  <div className="empty-state-desc">Configure the selection selectors from the parameters config panel on the left to run predictions.</div>
                </div>
              )}
            </div>

            {/* What-If Simulator Panel */}
            {forecastData && (
              <div className="glass-panel schedule-panel">
                <div className="schedule-header">
                  <div>
                    <span className="section-label">
                      <Sliders size={11} />
                      SIMULATOR
                    </span>
                    <h3 className="schedule-title">What-If Planner & Interactive Simulator</h3>
                  </div>
                  <button 
                    type="button" 
                    onClick={resetOverrides}
                    className="bg-transparent border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition-colors"
                  >
                    <RefreshCw size={11} /> Reset to Baseline
                  </button>
                </div>
                
                <p className="schedule-desc">
                  Modify the operating schedules and promotional campaign calendar below to simulate sales impacts.
                </p>

                <div className="schedule-grid">
                  {forecastData.predictions.map((p, idx) => {
                    const overrideItem = overrides.find(o => o.date === p.date) || p;
                    const dateObj = new Date(p.date);
                    const dayName = dateObj.toLocaleDateString('en-US', { weekday: 'short' });
                    
                    return (
                      <div key={p.date} className="simulator-card-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span className="schedule-date">
                            {dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                          </span>
                          <span className="schedule-day">{dayName}</span>
                        </div>
                        
                        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', margin: '4px 0' }}></div>
 
                        {/* Open toggle */}
                        <div className="simulator-toggle-row">
                          <span>Open</span>
                          <label className="switch">
                            <input 
                              type="checkbox"
                              checked={overrideItem.open === 1}
                              onChange={() => handleToggleOverride(p.date, "open")}
                              disabled={loading}
                            />
                            <span className="slider"></span>
                          </label>
                        </div>

                        {/* Promo toggle */}
                        <div className="simulator-toggle-row">
                          <span>Promo</span>
                          <label className="switch promo">
                            <input 
                              type="checkbox"
                              checked={overrideItem.open === 1 && overrideItem.promo === 1}
                              onChange={() => handleToggleOverride(p.date, "promo")}
                              disabled={loading || overrideItem.open === 0}
                            />
                            <span className="slider"></span>
                          </label>
                        </div>
                        
                        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', margin: '4px 0' }}></div>
                        
                        {/* Daily Forecast Sales Output */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', fontWeight: 'bold' }}>
                          <span className="text-slate-500">Sales:</span>
                          <span className={overrideItem.open === 0 ? 'text-rose-400' : 'text-indigo-400'}>
                            {overrideItem.open === 0 ? 'Closed' : `₹${Math.round(p.predicted_sales).toLocaleString()}`}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
