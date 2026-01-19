import React from 'react';

/**
 * TODO (Chapter 08):
 * 1. Accept props: { filters, onFilterChange }.
 * 2. Render three control groups:
 *    a. Checkbox → "Show only open seats" -> updates filters.showOnlyOpen.
 *    b. Segmented buttons or radio inputs for delivery mode (all, in-person, remote).
 *    c. Select dropdown for focus area (all, events, forms, assessment).
 * 3. Every control must be fully controlled (value comes from filters, changes call onFilterChange).
 * 4. Demonstrate conditional rendering by:
 *    - Showing helper text only when a filter is active.
 *    - Highlighting the active mode button.
 * 5. Optional: support a keyboard shortcut (e.g., press "f" to toggle the checkbox) via props passed from App.
 */
function FilterPanel({ filters, onFilterChange }) {
  return (
    <div className="filters-grid">
      {/* Checkbox for open seats only */}
      <div className="filter-group">
        <label>
          <input
            type="checkbox"
            checked={filters.showOnlyOpen}
            onChange={(e) => onFilterChange({ showOnlyOpen: e.target.checked })}
          />
          <span>Show only open seats</span>
        </label>
        {filters.showOnlyOpen && (
          <p className="filter-hint">Hiding full workshops</p>
        )}
      </div>

      {/* Delivery mode buttons */}
      <div className="filter-group">
        <label className="filter-label">Delivery Mode</label>
        <div className="mode-buttons">
          <button
            type="button"
            className={filters.mode === 'all' ? 'active' : ''}
            onClick={() => onFilterChange({ mode: 'all' })}
          >
            All
          </button>
          <button
            type="button"
            className={filters.mode === 'in-person' ? 'active' : ''}
            onClick={() => onFilterChange({ mode: 'in-person' })}
          >
            In-Person
          </button>
          <button
            type="button"
            className={filters.mode === 'remote' ? 'active' : ''}
            onClick={() => onFilterChange({ mode: 'remote' })}
          >
            Remote
          </button>
        </div>
      </div>

      {/* Focus area select */}
      <div className="filter-group">
        <label htmlFor="focus-select" className="filter-label">
          Focus Area
        </label>
        <select
          id="focus-select"
          value={filters.focus}
          onChange={(e) => onFilterChange({ focus: e.target.value })}
        >
          <option value="all">All Topics</option>
          <option value="events">Events</option>
          <option value="forms">Forms</option>
          <option value="assessment">Assessment</option>
        </select>
      </div>
    </div>
  );
}

export default FilterPanel;


