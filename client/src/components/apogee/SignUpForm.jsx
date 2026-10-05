import React, { useState } from 'react';
import { useT } from '../../context/LanguageContext';

export function SignUpForm({ isActive, onSuccess }) {
  const t = useT();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [termsAgree, setTermsAgree] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  // Live Password Strength Calculation
  const getScore = (val) => {
    let score = 0;
    if (val.length >= 6) score++;
    if (val.length >= 10) score++;
    if (/[A-Z]/.test(val) && /[0-9]/.test(val)) score++;
    if (/[^A-Za-z0-9]/.test(val)) score++;
    return score;
  };

  const score = getScore(password);
  const colors = ['rgba(152, 161, 199, 0.2)', '#ef4444', '#f59e0b', '#10b981', '#3fe7c8'];
  const labels = [
    t.strengthWeak,
    t.strengthWeak,
    t.strengthFair,
    t.strengthStrong,
    t.strengthCelestial,
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (loading || success) return;

    setLoading(true);

    setTimeout(() => {
      setLoading(false);
      setSuccess(true);

      setTimeout(() => {
        if (onSuccess) onSuccess();
      }, 1500);
    }, 1800);
  };

  return (
    <div
      className={`form-panel ${isActive ? 'active' : ''}`}
      id="panelSignup"
      role="tabpanel"
      aria-hidden={!isActive}
    >
      <form className="auth-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label" htmlFor="signupName">{t.commanderName}</label>
          <div className="input-wrapper">
            <input
              type="text"
              id="signupName"
              className="form-input"
              placeholder="Cmdr. Elena Rostova"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="name"
            />
            <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="signupEmail">{t.orbitalEmail}</label>
          <div className="input-wrapper">
            <input
              type="email"
              id="signupEmail"
              className="form-input"
              placeholder="commander@apogee.space"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />
            <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
              <polyline points="22,6 12,13 2,6" />
            </svg>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="signupPassword">{t.createSecurityKey}</label>
          <div className="input-wrapper">
            <input
              type={showPassword ? 'text' : 'password'}
              id="signupPassword"
              className="form-input"
              placeholder={t.atLeast8Chars}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
            />
            <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
            <button
              type="button"
              className="password-toggle"
              aria-label="Toggle password visibility"
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                  <line x1="1" y1="1" x2="23" y2="23" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              )}
            </button>
          </div>

          {/* Password Strength Indicator */}
          <div className="strength-meter">
            <div className="strength-bars">
              <div className="strength-bar-step" style={{ backgroundColor: score >= 1 ? colors[score] : colors[0] }} />
              <div className="strength-bar-step" style={{ backgroundColor: score >= 2 ? colors[score] : colors[0] }} />
              <div className="strength-bar-step" style={{ backgroundColor: score >= 3 ? colors[score] : colors[0] }} />
              <div className="strength-bar-step" style={{ backgroundColor: score >= 4 ? colors[score] : colors[0] }} />
            </div>
            <span className="strength-text" style={{ color: score > 0 ? colors[score] : 'var(--color-muted)' }}>
              {password ? labels[score] : t.strengthWeak}
            </span>
          </div>
        </div>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              className="checkbox-input"
              required
              checked={termsAgree}
              onChange={(e) => setTermsAgree(e.target.checked)}
            />
            <span className="custom-checkbox">
              <svg className="checkbox-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </span>
            <span style={{ fontSize: '0.82rem' }}>
              {t.agreeTerms} <a href="#terms" className="forgot-link" onClick={(e) => e.preventDefault()}>{t.orbitalCharter}</a> {t.privacyProtocol}
            </span>
          </label>
        </div>

        <button
          type="submit"
          className={`submit-btn ${loading ? 'loading' : ''} ${success ? 'success' : ''}`}
          disabled={loading || success}
        >
          <span className="btn-spinner" aria-hidden="true" />
          <span className="btn-text">
            {success ? t.accountCreated : loading ? t.provisioning : t.createExplorerAccount}
          </span>
        </button>
      </form>
    </div>
  );
}

export default SignUpForm;
