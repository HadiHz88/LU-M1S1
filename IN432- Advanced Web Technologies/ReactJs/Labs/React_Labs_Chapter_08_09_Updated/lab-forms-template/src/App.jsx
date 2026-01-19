import React from 'react';

// TODO: Import useState from React

// Course options for the dropdown
const COURSES = [
  { value: '', label: 'Select a course...' },
  { value: 'react', label: 'React Fundamentals' },
  { value: 'javascript', label: 'JavaScript Basics' },
  { value: 'css', label: 'CSS & Styling' }
];

function App() {
  // TODO (Part 1): Create state for name and email
  // const [name, setName] = useState('')
  // const [email, setEmail] = useState('')

  // TODO (Part 2): Create state for course selection
  // const [course, setCourse] = useState('')

  // TODO (Part 3): Create state for star rating
  // const [rating, setRating] = useState(0)

  // TODO (Part 4): Create state for feedback text
  // const [feedback, setFeedback] = useState('')

  // TODO (Part 5): Create state for validation errors
  // const [errors, setErrors] = useState({})

  // TODO (Part 6): Create state for submission
  // const [submitted, setSubmitted] = useState(false)

  // TODO (Part 5): Create validate() function
  // - Check name is not empty and at least 2 chars
  // - Check email contains '@'
  // - Check course is selected
  // - Check rating > 0
  // - Return object with error messages

  // TODO (Part 6): Create handleSubmit function
  // - Call validate()
  // - If errors exist, save to state and return
  // - If valid, set submitted to true

  // TODO (Part 7): Create handleClear function
  // - Reset all state values to initial

  // TODO (Part 6): Create handleSubmitAnother function
  // - Call handleClear()
  // - Set submitted to false

  // TODO (Part 6): Show Thank You screen when submitted
  // if (submitted) {
  //   return (
  //     <div className="form-container">
  //       <div className="thank-you">
  //         <div className="checkmark">✓</div>
  //         <h2>Thank You!</h2>
  //         <div className="summary">
  //           <p><strong>Name:</strong> {name}</p>
  //           <p><strong>Email:</strong> {email}</p>
  //           {/* ... more fields ... */}
  //         </div>
  //         <button onClick={handleSubmitAnother}>Submit Another</button>
  //       </div>
  //     </div>
  //   )
  // }

  return (
    <div className="form-container">
      <h1 className="form-title">📝 Student Feedback Form</h1>

      {/* Part 1: Name Input */}
      <div className="form-group">
        <label>Full Name *</label>
        <input 
          type="text" 
          placeholder="Enter your name"
          // TODO: Add value={name} and onChange={(e) => setName(e.target.value)}
        />
        {/* TODO: Show error - {errors.name && <p className="error-text">{errors.name}</p>} */}
      </div>

      {/* Part 1: Email Input */}
      <div className="form-group">
        <label>Email *</label>
        <input 
          type="email" 
          placeholder="Enter your email"
          // TODO: Add value and onChange
        />
        {/* TODO: Show error for email */}
      </div>

      {/* Part 2: Course Dropdown */}
      <div className="form-group">
        <label>Course *</label>
        <select>
          {/* TODO: Add value={course} and onChange={(e) => setCourse(e.target.value)} */}
          {COURSES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
        {/* TODO: Show error for course */}
      </div>

      {/* Part 3: Star Rating */}
      <div className="form-group">
        <label>Rating *</label>
        <div className="star-rating">
          {/* TODO: Create 5 star buttons with onClick */}
          {/* Example for one star: */}
          {/* <button 
               type="button"
               className={`star-btn ${1 <= rating ? 'filled' : ''}`}
               onClick={() => setRating(1)}
             >
               {1 <= rating ? '★' : '☆'}
             </button> 
          */}
          {[1, 2, 3, 4, 5].map((star) => (
            <button key={star} type="button" className="star-btn">
              ☆
            </button>
          ))}
        </div>
        <p className="rating-text">Click to rate</p>
        {/* TODO: Show "You rated: X/5 stars" when rating > 0 */}
        {/* TODO: Show error for rating */}
      </div>

      {/* Part 4: Feedback Textarea */}
      <div className="form-group">
        <label>Your Feedback (optional)</label>
        <textarea 
          placeholder="Tell us what you think..."
          rows={4}
          // TODO: Add value and onChange
        />
        <p className="char-count">0 / 500 characters</p>
        {/* TODO: Update character count dynamically */}
        {/* TODO: Change class to 'char-count warning' at 400 chars */}
        {/* TODO: Change class to 'char-count danger' at 500 chars */}
      </div>

      {/* Part 6 & 7: Buttons */}
      <div className="button-group">
        {/* TODO (Part 7): Show Clear button only when form has data */}
        <button type="button" className="btn btn-secondary">
          Clear Form
        </button>
        {/* TODO (Part 6): Add onClick={handleSubmit} */}
        <button type="button" className="btn btn-primary">
          Submit Feedback
        </button>
      </div>
    </div>
  );
}

export default App;
