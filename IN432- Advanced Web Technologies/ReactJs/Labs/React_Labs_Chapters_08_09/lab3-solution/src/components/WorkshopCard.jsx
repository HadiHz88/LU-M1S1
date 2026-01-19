import React from 'react';

/**
 * TODO (Chapter 08):
 * 1. Accept props using destructuring: { workshop, isSelected, onSelect }.
 * 2. Compute derived booleans:
 *    - isFull when seats.taken >= seats.total.
 *    - isAlmostFull when remaining seats <= 3.
 * 3. Apply conditional class names:
 *    - Always include "workshop-card".
 *    - Add "workshop-card--selected" when isSelected is true.
 *    - Add "workshop-card--full" when isFull is true.
 * 4. Render the workshop info:
 *    - Title, instructor, mode, session (remember to keep ق.ظ./ب.ظ.).
 *    - Remaining seats (or "Waitlist only" when full).
 *    - Tags mapped into <span className="pill"> elements.
 * 5. Emit onSelect(workshop.id) when the card/button is clicked.
 * 6. Stretch: show a different emoji/icon for in-person vs remote sessions.
 */
function WorkshopCard({ workshop, isSelected, onSelect }) {
  const isFull = workshop.seats.taken >= workshop.seats.total;
  const remainingSeats = workshop.seats.total - workshop.seats.taken;
  const isAlmostFull = remainingSeats > 0 && remainingSeats <= 3;

  const cardClass = [
    'workshop-card',
    isSelected && 'workshop-card--selected',
    isFull && 'workshop-card--full'
  ].filter(Boolean).join(' ');

  const getSeatStatus = () => {
    if (isFull) return { text: 'Waitlist only', className: 'pill--danger' };
    if (isAlmostFull) return { text: 'Filling fast', className: 'pill--warning' };
    return { text: 'Open', className: 'pill--success' };
  };

  const seatStatus = getSeatStatus();
  const modeIcon = workshop.mode === 'remote' ? '🌐' : '📍';

  return (
    <article className={cardClass} onClick={() => onSelect(workshop.id)}>
      <header>
        <p className="eyebrow">{workshop.focus}</p>
        <h3>{workshop.title}</h3>
      </header>

      <div className="workshop-meta">
        <p className="workshop-instructor">👤 {workshop.instructor}</p>
        <p className="workshop-mode">{modeIcon} {workshop.mode}</p>
        <p className="workshop-session">🕐 {workshop.session}</p>
        <p className="workshop-duration">⏱️ {workshop.duration}</p>
        <p className="workshop-level">📊 {workshop.level}</p>
      </div>

      <p className="workshop-summary">{workshop.summary}</p>

      <div className="workshop-badges">
        <span className={`pill ${seatStatus.className}`}>
          {seatStatus.text}
        </span>
        {!isFull && (
          <span className="pill">
            {remainingSeats} / {workshop.seats.total} seats
          </span>
        )}
      </div>

      <div className="workshop-tags">
        {workshop.tags.map(tag => (
          <span key={tag} className="pill pill--small">{tag}</span>
        ))}
      </div>

      <button
        type="button"
        className="workshop-select-btn"
        onClick={(e) => {
          e.stopPropagation();
          onSelect(workshop.id);
        }}
      >
        {isSelected ? 'Selected' : 'Select Workshop'}
      </button>
    </article>
  );
}

export default WorkshopCard;


