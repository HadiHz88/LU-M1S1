import React, { useState } from 'react';

// Course options for the dropdown
const COURSES = [
  { value: '', label: 'Select a course...' },
  { value: 'react', label: 'React Fundamentals' },
  { value: 'javascript', label: 'JavaScript Basics' },
  { value: 'css', label: 'CSS & Styling' }
];

function App() {
  // State management
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [course, setCourse] = useState('');
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(false);

  // Validation function
  const validate = () => {
    const newErrors = {};
    
    if (name.trim().length < 2) {
      newErrors.name = 'Name must be at least 2 characters';
    }
    
    if (!email.includes('@')) {
      newErrors.email = 'Please enter a valid email';
    }
    
    if (!course) {
      newErrors.course = 'Please select a course';
    }
    
    if (rating === 0) {
      newErrors.rating = 'Please provide a rating';
    }
    
    return newErrors;
  };

  // Handle form submission
  const handleSubmit = () => {
    const validationErrors = validate();
    
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    
    // Clear errors and submit
    setErrors({});
    setSubmitted(true);
  };

  // Clear form
  const handleClear = () => {
    setName('');
    setEmail('');
    setCourse('');
    setRating(0);
    setFeedback('');
    setErrors({});
  };

  // Submit another response
  const handleSubmitAnother = () => {
    handleClear();
    setSubmitted(false);
  };

  // Check if form has data
  const hasFormData = name || email || course || rating > 0 || feedback;

  // Get character count class
  const getCharCountClass = () => {
    const length = feedback.length;
    if (length >= 500) return 'char-count danger';
    if (length >= 400) return 'char-count warning';
    return 'char-count';
  };

  // Get course label from value
  const getCourseLabel = () => {
    const selected = COURSES.find(c => c.value === course);
    return selected ? selected.label : '';
  };

  // Thank You screen
  if (submitted) {
    return (
      <div className="form-container">
        <div className="thank-you">
          <div className="checkmark">✓</div>
          <h2>Thank You!</h2>
          <p className="thank-you-subtitle">Your feedback has been submitted successfully.</p>
          
          <div className="summary">
            <div className="summary-item">
              <strong>Name:</strong>
              <span>{name}</span>
            </div>
            
            <div className="summary-item">
              <strong>Email:</strong>
              <span>{email}</span>
            </div>
            
            <div className="summary-item">
              <strong>Course:</strong>
              <span>{getCourseLabel()}</span>
            </div>
            
            <div className="summary-item">
              <strong>Rating:</strong>
              <span>
                {[...Array(5)].map((_, i) => (
                  <span key={i} className="summary-star">
                    {i < rating ? '★' : '☆'}
                  </span>
                ))}
                ({rating}/5 stars)
              </span>
            </div>
            
            {feedback && (
              <div className="summary-item">
                <strong>Feedback:</strong>
                <span className="feedback-text">{feedback}</span>
              </div>
            )}
          </div>
          
          <button className="btn btn-primary" onClick={handleSubmitAnother}>
            Submit Another Feedback
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="form-container">
      <h1 className="form-title">📝 Student Feedback Form</h1>

      {/* Part 1: Name Input */}
      <div className="form-group">
        <label>Full Name *</label>
        <input 
          type="text" 
          placeholder="Enter your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        {errors.name && <p className="error-text">{errors.name}</p>}
      </div>

      {/* Part 1: Email Input */}
      <div className="form-group">
        <label>Email *</label>
        <input 
          type="email" 
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        {errors.email && <p className="error-text">{errors.email}</p>}
      </div>

      {/* Part 2: Course Dropdown */}
      <div className="form-group">
        <label>Course *</label>
        <select value={course} onChange={(e) => setCourse(e.target.value)}>
          {COURSES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
        {errors.course && <p className="error-text">{errors.course}</p>}
      </div>

      {/* Part 3: Star Rating */}
      <div className="form-group">
        <label>Rating *</label>
        <div className="star-rating">
          {[1, 2, 3, 4, 5].map((star) => (
            <button 
              key={star} 
              type="button" 
              className={`star-btn ${star <= rating ? 'filled' : ''}`}
              onClick={() => setRating(star)}
            >
              {star <= rating ? '★' : '☆'}
            </button>
          ))}
        </div>
        {rating > 0 ? (
          <p className="rating-text">You rated: {rating}/5 stars</p>
        ) : (
          <p className="rating-text">Click to rate</p>
        )}
        {errors.rating && <p className="error-text">{errors.rating}</p>}
      </div>

      {/* Part 4: Feedback Textarea */}
      <div className="form-group">
        <label>Your Feedback (optional)</label>
        <textarea 
          placeholder="Tell us what you think..."
          rows={4}
          value={feedback}
          onChange={(e) => setFeedback(e.target.value.slice(0, 500))}
        />
        <p className={getCharCountClass()}>
          {feedback.length} / 500 characters
        </p>
      </div>

      {/* Part 6 & 7: Buttons */}
      <div className="button-group">
        {hasFormData && (
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={handleClear}
          >
            Clear Form
          </button>
        )}
        <button 
          type="button" 
          className="btn btn-primary"
          onClick={handleSubmit}
        >
          Submit Feedback
        </button>
      </div>
    </div>
  );
}

export default App;
