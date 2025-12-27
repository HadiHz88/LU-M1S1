/*
Student ID: 105174
Full Name: Hadi Hijazi
*/

import React, { useState } from 'react'
import './App.css'

// Question 1: StudentCard Component
function StudentCard({ name, studentId, course }) {
  return (
    <div style={{ border: '1px solid #ccc', padding: '10px', margin: '10px', borderRadius: '5px' }}>
      <h3>{name}</h3>
      <p><strong>Student ID:</strong> {studentId}</p>
      <p><strong>Course:</strong> {course}</p>
    </div>
  )
}

// Question 2: Product Component
function Product({ name, price, inStock }) {
  return (
    <div style={{ border: '1px solid #ccc', padding: '10px', margin: '10px', borderRadius: '5px' }}>
      <h3>{name}</h3>
      <p><strong>Price:</strong> ${price}</p>
      {inStock ? (
        <p style={{ color: 'green' }}>In Stock</p>
      ) : (
        <p style={{ color: 'red' }}>Out of Stock</p>
      )}
    </div>
  )
}

// Question 3: Counter Component
function Counter() {
  const [count, setCount] = useState(0)

  const increment = () => {
    setCount(count + 1)
  }

  const decrement = () => {
    setCount(count - 1)
  }

  return (
    <div style={{ padding: '10px', margin: '10px' }}>
      <h3>Count: {count}</h3>
      <button onClick={increment} style={{ margin: '5px', padding: '5px 15px' }}>Increment</button>
      <button onClick={decrement} style={{ margin: '5px', padding: '5px 15px' }}>Decrement</button>
    </div>
  )
}

// Question 4: NameCounter Component
function NameCounter() {
  const [count, setCount] = useState(0)
  const [name, setName] = useState("Counter")

  const incrementCount = () => {
    setCount(count + 1)
  }

  const changeName = () => {
    setName("Updated Counter")
  }

  return (
    <div style={{ padding: '10px', margin: '10px' }}>
      <h3>{name}: {count}</h3>
      <button onClick={incrementCount} style={{ margin: '5px', padding: '5px 15px' }}>Increment</button>
      <button onClick={changeName} style={{ margin: '5px', padding: '5px 15px' }}>Change Name</button>
    </div>
  )
}

// Question 5: ItemList Component
function ItemList() {
  const [items, setItems] = useState([])
  const [inputValue, setInputValue] = useState("")

  const handleAdd = () => {
    if (inputValue.trim() !== "") {
      setItems([...items, inputValue])
      setInputValue("")
    }
  }

  return (
    <div style={{ padding: '10px', margin: '10px' }}>
      <input 
        type="text" 
        value={inputValue} 
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Enter item"
        style={{ padding: '5px', margin: '5px' }}
      />
      <button onClick={handleAdd} style={{ margin: '5px', padding: '5px 15px' }}>Add</button>
      <ul>
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

// Question 6: RegistrationForm Component
function RegistrationForm() {
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [submittedData, setSubmittedData] = useState(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    setSubmittedData({ name, email, password })
  }

  return (
    <div style={{ padding: '10px', margin: '10px' }}>
      <form onSubmit={handleSubmit}>
        <div style={{ margin: '10px 0' }}>
          <label>
            Name: 
            <input 
              type="text" 
              value={name} 
              onChange={(e) => setName(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px' }}
            />
          </label>
        </div>
        <div style={{ margin: '10px 0' }}>
          <label>
            Email: 
            <input 
              type="email" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px' }}
            />
          </label>
        </div>
        <div style={{ margin: '10px 0' }}>
          <label>
            Password: 
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px' }}
            />
          </label>
        </div>
        <button type="submit" style={{ padding: '5px 15px', marginTop: '10px' }}>Submit</button>
      </form>

      {submittedData && (
        <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}>
          <h4>Submitted Data:</h4>
          <p><strong>Name:</strong> {submittedData.name}</p>
          <p><strong>Email:</strong> {submittedData.email}</p>
          <p><strong>Password:</strong> {submittedData.password}</p>
        </div>
      )}
    </div>
  )
}

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
            <StudentCard name="Hadi Hijazi" studentId="105174" course="Advanced Web Technologies" />
          </section>

          <section className="question-section">
            <h2>Question 2: Product Component</h2>
            <Product name="Laptop" price={999.99} inStock={true} />
            <Product name="Smartphone" price={699.99} inStock={false} />
          </section>

          <section className="question-section">
            <h2>Question 3: Counter Component</h2>
            <Counter />
          </section>

          <section className="question-section">
            <h2>Question 4: NameCounter Component (useState)</h2>
            <NameCounter />
          </section>

          <section className="question-section">
            <h2>Question 5: ItemList Component (Events)</h2>
            <ItemList />
          </section>

          <section className="question-section">
            <h2>Question 6: RegistrationForm Component</h2>
            <RegistrationForm />
          </section>
        </div>
      </header>
    </div>
  )
}

export default App
