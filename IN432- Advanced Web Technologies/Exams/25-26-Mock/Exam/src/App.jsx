/*
Student ID: [Your Student ID]
Full Name: [Your Full Name]
*/

import React from 'react'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>React Mock Exam</h1>
        <p>Answer the questions in exam.md</p>
        <p>Start implementing your solutions below:</p>
        
        {/* Your solutions go here */}
        <div className="solutions">
          <section className="question-section">
            <h2>Question 1: StudentCard Component</h2>
            {/* Add your StudentCard component here */}
          </section>

          <section className="question-section">
            <h2>Question 2: Product Component</h2>
            {/* Add your Product component here */}
          </section>

          <section className="question-section">
            <h2>Question 3: Counter Component</h2>
            {/* Add your Counter component here */}
          </section>

          <section className="question-section">
            <h2>Question 4: NameCounter Component (useState)</h2>
            {/* Add your NameCounter component here */}
          </section>

          <section className="question-section">
            <h2>Question 5: ItemList Component (Events)</h2>
            {/* Add your ItemList component here */}
          </section>

          <section className="question-section">
            <h2>Question 6: RegistrationForm Component</h2>
            {/* Add your RegistrationForm component here */}
          </section>
        </div>
      </header>
    </div>
  )
}

export default App
