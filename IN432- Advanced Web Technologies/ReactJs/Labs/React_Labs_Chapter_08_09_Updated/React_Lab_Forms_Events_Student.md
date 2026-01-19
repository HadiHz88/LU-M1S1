## React Lab – Forms & Events (Student Guide)

**Goal**  
Build a simple Student Feedback Form to practice controlled inputs, `onChange` events, and `onClick` handlers using only `useState`.

**Template Requirement**  

1. Duplicate the provided `lab-forms-template` folder and rename it `react-forms-<yourname>`.  
2. Run `npm install` inside the duplicated folder.  
3. Do **not** rename starter files.

**Styling Rules**  

1. The file `src/styles/app.css` contains basic styling.  
2. Feel free to add your own styles to make it look nice!

---

### Part 1 – Text Inputs with onChange

1. In `App.jsx`, import `useState` and create state for:

   ```javascript
   const [name, setName] = useState('')
   const [email, setEmail] = useState('')
   ```

2. Create two text inputs:
   - One for "Full Name" that updates `name` on change
   - One for "Email" that updates `email` on change

3. Use the `onChange` event to update state:

   ```javascript
   <input 
     type="text" 
     value={name} 
     onChange={(e) => setName(e.target.value)} 
   />
   ```

4. Display the current values below the inputs to verify they work.

---

### Part 2 – Select Dropdown with onChange

1. Add state for the selected course:

   ```javascript
   const [course, setCourse] = useState('')
   ```

2. Create a `<select>` dropdown with these options:
   - "" (empty - "Select a course...")
   - "react" - "React Fundamentals"
   - "javascript" - "JavaScript Basics"
   - "css" - "CSS & Styling"

3. Use `onChange` to update the `course` state when user selects an option.

4. Show the selected course name below the dropdown.

---

### Part 3 – Star Rating with onClick

1. Add state for the rating:

   ```javascript
   const [rating, setRating] = useState(0)
   ```

2. Create 5 star buttons (you can use ★ and ☆ characters):
   - When clicked, set the rating to that star's number (1-5)
   - Filled stars (★) for values ≤ current rating
   - Empty stars (☆) for values > current rating

3. Use `onClick` to update the rating:

   ```javascript
   <button onClick={() => setRating(3)}>★</button>
   ```

4. Display "You rated: X/5 stars" below the buttons.

---

### Part 4 – Textarea with onChange

1. Add state for feedback text:

   ```javascript
   const [feedback, setFeedback] = useState('')
   ```

2. Create a `<textarea>` for detailed feedback.

3. Show a character count: "X / 500 characters"

4. Optionally change the counter color when approaching the limit.

---

### Part 5 – Field Validation

1. Add state for error messages:

   ```javascript
   const [errors, setErrors] = useState({})
   ```

2. Create a `validate()` function that checks:
   - **Name**: Must not be empty, at least 2 characters
   - **Email**: Must not be empty, must contain `@`
   - **Course**: Must be selected (not empty string)
   - **Rating**: Must be greater than 0

3. Return an object with error messages:

   ```javascript
   function validate() {
     const newErrors = {}
     if (name.trim().length < 2) {
       newErrors.name = 'Name must be at least 2 characters'
     }
     if (!email.includes('@')) {
       newErrors.email = 'Please enter a valid email'
     }
     // ... more validations
     return newErrors
   }
   ```

4. Display error messages below each field in red.

---

### Part 6 – Form Submission & Thank You Screen

1. Add state to track submission:

   ```javascript
   const [submitted, setSubmitted] = useState(false)
   ```

2. Create a Submit button with `onClick` handler that:
   - Calls `validate()` to get errors
   - If there are errors: save them to state and stop
   - If valid: set `submitted` to `true`

3. When `submitted` is `true`:
   - Hide the form
   - Show a "Thank You" message with all submitted info
   - Display: Name, Email, Course, Rating (with stars), Feedback

4. Add a "Submit Another" button that:
   - Resets all form fields
   - Sets `submitted` back to `false`

---

### Part 7 – Clear Form Button (Bonus)

1. Add a "Clear Form" button next to Submit.

2. When clicked, reset all state values to their initial values.

3. This button should only be visible when the form has some data.

---

### Example Final Layout

```
┌─────────────────────────────────────┐
│     📝 Student Feedback Form        │
├─────────────────────────────────────┤
│  Full Name: [__________________]    │
│  Email:     [__________________]    │
│                                     │
│  Course:    [Select a course... ▼]  │
│                                     │
│  Rating:    ★ ★ ★ ☆ ☆              │
│             You rated: 3/5 stars    │
│                                     │
│  Feedback:                          │
│  ┌─────────────────────────────┐   │
│  │                             │   │
│  │                             │   │
│  └─────────────────────────────┘   │
│  45 / 500 characters                │
│                                     │
│  [Clear Form]  [Submit Feedback]    │
└─────────────────────────────────────┘
```

---

### Deliverables

1. Working React project with all form controls.
2. All inputs are controlled (values come from state).
3. Form validates before submission.
4. Thank you screen shows after successful submit.

### Submission Checklist

- [ ] Template duplicated, dependencies installed.
- [ ] Text inputs update state with `onChange`.
- [ ] Dropdown updates state with `onChange`.
- [ ] Star rating works with `onClick`.
- [ ] Textarea shows character count.
- [ ] Validation errors display below each field.
- [ ] Submit button validates and shows thank you screen with all info.
- [ ] Form resets after successful submission.
- [ ] Clear button resets the form.

Good luck! This lab focuses on the basics – take your time to understand how `onChange` and `onClick` connect your UI to React state.
