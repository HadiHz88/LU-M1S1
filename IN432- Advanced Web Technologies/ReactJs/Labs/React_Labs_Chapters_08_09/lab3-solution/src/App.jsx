import React, { useState, useEffect } from 'react';
import { INITIAL_WORKSHOPS } from './data/workshops';
import FilterPanel from './components/FilterPanel';
import WorkshopCard from './components/WorkshopCard';
import RegistrationForm from './components/RegistrationForm';
import SummaryPanel from './components/SummaryPanel';

// React Lab 03 focuses on Chapter 08 (Conditional Rendering) and Chapter 09 (Forms).
// Follow the inline TODOs to wire up the Workshop Registration Control Room.

function App() {
  // State management
  const [workshops, setWorkshops] = useState(INITIAL_WORKSHOPS);
  const [filters, setFilters] = useState({
    showOnlyOpen: false,
    mode: 'all',
    focus: 'all'
  });
  const [selectedWorkshopId, setSelectedWorkshopId] = useState(INITIAL_WORKSHOPS[0]?.id);
  const [lastRegistration, setLastRegistration] = useState(null);

  // Derive filteredWorkshops based on filters
  const filteredWorkshops = workshops.filter(workshop => {
    // Filter by open seats
    if (filters.showOnlyOpen && workshop.seats.taken >= workshop.seats.total) {
      return false;
    }
    // Filter by mode
    if (filters.mode !== 'all' && workshop.mode !== filters.mode) {
      return false;
    }
    // Filter by focus
    if (filters.focus !== 'all' && workshop.focus !== filters.focus) {
      return false;
    }
    return true;
  });

  // Compute selectedWorkshop
  const selectedWorkshop = workshops.find(w => w.id === selectedWorkshopId);

  // Auto-fallback if selected workshop is filtered out
  useEffect(() => {
    if (selectedWorkshop && !filteredWorkshops.find(w => w.id === selectedWorkshopId)) {
      if (filteredWorkshops.length > 0) {
        setSelectedWorkshopId(filteredWorkshops[0].id);
      }
    }
  }, [filteredWorkshops, selectedWorkshopId, selectedWorkshop]);

  // Helper values
  const openCount = workshops.filter(w => w.seats.taken < w.seats.total).length;
  const totalCount = workshops.length;
  const isFull = selectedWorkshop ? selectedWorkshop.seats.taken >= selectedWorkshop.seats.total : false;
  const remainingSeats = selectedWorkshop ? selectedWorkshop.seats.total - selectedWorkshop.seats.taken : 0;
  const almostFull = selectedWorkshop && remainingSeats > 0 && remainingSeats <= 3;
  const noResults = filteredWorkshops.length === 0;

  // Build filter summary caption
  const getFilterSummary = () => {
    const parts = [];
    if (filters.showOnlyOpen) parts.push('open sessions');
    if (filters.mode !== 'all') parts.push(`${filters.mode}-only workshops`);
    if (filters.focus !== 'all') parts.push(`focused on ${filters.focus}`);
    
    if (parts.length === 0) return null;
    return `Showing ${parts.join(', ')}`;
  };

  // Handlers
  const handleFilterChange = (partial) => {
    setFilters(prev => ({ ...prev, ...partial }));
  };

  const handleSelectWorkshop = (id) => {
    setSelectedWorkshopId(id);
  };

  const handleRegistrationSubmit = (payload) => {
    const workshop = workshops.find(w => w.id === payload.workshopId);
    if (!workshop) return;

    const hasSeats = workshop.seats.taken < workshop.seats.total;
    const isWaitlist = !hasSeats;

    // Update seats if available
    if (hasSeats) {
      setWorkshops(prev => prev.map(w => 
        w.id === payload.workshopId
          ? { ...w, seats: { ...w.seats, taken: w.seats.taken + 1 } }
          : w
      ));
    }

    // Store registration
    setLastRegistration({
      ...payload,
      waitlist: isWaitlist,
      submittedAt: new Date().toISOString(),
      workshopTitle: workshop.title,
      workshopSession: workshop.session
    });
  };

  return (
    <main className="lab-shell">
      <header className="lab-hero card">
        <div>
          <p className="eyebrow">React Lab 03 · Chapters 08–09</p>
          <h1>Workshop Registration Control Room</h1>
          <p className="intro">
            Practice conditional rendering patterns and controlled forms by managing the workshop schedule below.
          </p>
        </div>

        <div className="hero-stats">
          <span className="pill">
            {openCount} of {totalCount} sessions still open
          </span>
          {almostFull && (
            <span className="pill pill--warning">
              Urgent · Almost full
            </span>
          )}
        </div>
      </header>

      <section className="card filters-panel">
        <div className="filters-heading">
          <h2>Filter the Schedule</h2>
          <p>Use these controls to test dynamic UI rendering based on Chapter 08 techniques.</p>
        </div>

        <FilterPanel 
          filters={filters}
          onFilterChange={handleFilterChange}
        />

        {getFilterSummary() && (
          <p className="filter-caption">
            {getFilterSummary()}
          </p>
        )}
      </section>

      <section className="content-grid">
        <section className="card workshop-panel">
          <header className="panel-heading">
            <div>
              <p className="eyebrow">Sessions</p>
              <h2>Available Workshops</h2>
            </div>
            {selectedWorkshop && (
              <span className="pill pill--selected">{selectedWorkshop.title}</span>
            )}
          </header>

          {noResults ? (
            <article className="empty-state">
              <h3>No workshops match your filters</h3>
              <p>Try adjusting the filters above to see more options.</p>
            </article>
          ) : (
            <div className="workshop-grid">
              {filteredWorkshops.map(workshop => (
                <WorkshopCard
                  key={workshop.id}
                  workshop={workshop}
                  isSelected={workshop.id === selectedWorkshopId}
                  onSelect={handleSelectWorkshop}
                />
              ))}
            </div>
          )}
        </section>

        <section className="card form-panel">
          <header className="panel-heading">
            <div>
              <p className="eyebrow">Registration</p>
              <h2>Reserve Your Seat</h2>
            </div>
            {isFull && (
              <span className="pill pill--danger">Full · Waitlist only</span>
            )}
          </header>

          {selectedWorkshop && (
            <RegistrationForm
              workshop={selectedWorkshop}
              onSubmit={handleRegistrationSubmit}
            />
          )}

          {lastRegistration && (
            <SummaryPanel
              registration={lastRegistration}
              onClear={() => setLastRegistration(null)}
            />
          )}
        </section>
      </section>
    </main>
  );
}

export default App;

