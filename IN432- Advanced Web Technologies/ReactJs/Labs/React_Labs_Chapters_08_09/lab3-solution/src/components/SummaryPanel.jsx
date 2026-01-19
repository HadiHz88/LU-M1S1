import React from 'react';

/**
 * TODO (Chapter 08 + 09):
 * 1. Accept props: { registration, onClear }.
 * 2. When registration is null, render nothing (App will control via conditional rendering).
 * 3. Show a confirmation card containing:
 *    - Attendee name + email.
 *    - Target workshop title + session time.
 *    - Whether the attendee has a confirmed seat or a waitlist slot.
 *    - Any dietary/equipment notes (only if provided).
 * 4. Add a button that calls onClear() so the form can be used again.
 * 5. Style the banner differently for "confirmed" vs "waitlist" using CSS helper classes.
 */
function SummaryPanel({ registration, onClear }) {
  if (!registration) return null;

  const panelClass = registration.waitlist 
    ? 'summary-panel summary-panel--waitlist'
    : 'summary-panel summary-panel--confirmed';

  return (
    <section className={panelClass}>
      <header className="summary-header">
        <h3>
          {registration.waitlist ? '📋 Waitlist Registration' : '✅ Registration Confirmed'}
        </h3>
        <button 
          type="button" 
          className="clear-btn"
          onClick={onClear}
          aria-label="Clear summary"
        >
          ✕
        </button>
      </header>

      <div className="summary-content">
        <div className="summary-section">
          <h4>Attendee Information</h4>
          <p><strong>Name:</strong> {registration.fullName}</p>
          <p><strong>Email:</strong> {registration.email}</p>
          <p><strong>Experience:</strong> {registration.experience}</p>
          <p><strong>Attendance:</strong> {registration.attendanceType}</p>
        </div>

        <div className="summary-section">
          <h4>Workshop Details</h4>
          <p><strong>Workshop:</strong> {registration.workshopTitle}</p>
          <p><strong>Session:</strong> {registration.workshopSession}</p>
          <p>
            <strong>Status:</strong>{' '}
            <span className={registration.waitlist ? 'status-waitlist' : 'status-confirmed'}>
              {registration.waitlist ? 'Waitlist' : 'Confirmed Seat'}
            </span>
          </p>
        </div>

        {(registration.needsEquipment || registration.dietary !== 'none' || registration.notes) && (
          <div className="summary-section">
            <h4>Additional Information</h4>
            {registration.needsEquipment && (
              <p><strong>Equipment needed:</strong> {registration.equipmentDetails || 'Yes'}</p>
            )}
            {registration.dietary !== 'none' && (
              <p><strong>Dietary restrictions:</strong> {registration.dietary}</p>
            )}
            {registration.notes && (
              <p><strong>Notes:</strong> {registration.notes}</p>
            )}
          </div>
        )}

        <div className="summary-footer">
          <p className="timestamp">
            Submitted: {new Date(registration.submittedAt).toLocaleString()}
          </p>
          <button 
            type="button" 
            className="clear-summary-btn"
            onClick={onClear}
          >
            Clear Summary
          </button>
        </div>
      </div>
    </section>
  );
}

export default SummaryPanel;


