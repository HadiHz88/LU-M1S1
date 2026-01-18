# 📚 Node.js + Express + MongoDB Exam

## Library Management System API

**Duration:** 90 minutes  
**Total Points:** 100 points (+ 10 bonus)  
**Date:** January 18, 2026

---

## 📋 Instructions

### General Rules

- **No authentication required** - Focus on CRUD operations and queries
- **No external packages** except `express` and `mongoose`
- Use `async/await` for all database operations
- Follow REST API conventions
- Include proper error handling with try-catch blocks
- Test your endpoints using Postman, Thunder Client, or similar tools

### Submission Requirements

- Submit a single `.js` file (e.g., `server.js` or `app.js`)
- Include your **Student ID** and **Full Name** at the top as a comment
- Ensure your code runs without errors on `http://localhost:3000`
- Database name: `library_db`

### Grading Criteria

- **Functionality** (60%): Does it work as specified?
- **Code Quality** (25%): Clean, readable, well-structured code
- **Error Handling** (15%): Proper status codes and error messages

---

## 🎯 Scenario

You are building a **Library Management System API** that manages books and authors. The system should:

- Track books and their authors
- Allow librarians to manage inventory
- Provide search and filtering capabilities
- Track book availability (borrowed/available)

---

## 📊 Data Models

You must implement **two Mongoose schemas**: `Author` and `Book`.

**Required fields will be specified in each question.** Design your schemas to support all the operations described below.

---

## 📝 Questions

### **Question 1: Project Setup & Basic CRUD** (25 points)

Set up your Express + MongoDB project and implement basic operations.

**Setup Requirements:**

- Express server on port `3000`
- MongoDB connection to `mongodb://localhost:27017/library_db`
- Include timestamps for both models

**Author Model Fields:** `name` (required), `email` (required, unique), `country` (optional), `booksWritten` (optional)

**Book Model Fields:** `title` (required), `authorId` (required, references Author), `genre` (required), `publishedYear` (required), `pages` (required), `isAvailable` (boolean), `borrowedBy` (string)

**Implement these endpoints:**

- `POST /api/authors` - Create author (201)
- `GET /api/authors` - Get all authors (200)
- `GET /api/authors/:id` - Get author by ID (200 or 404)
- `POST /api/books` - Create book, verify authorId exists (201)
- `GET /api/books` - Get all books (200)

---

### **Question 2: Advanced Queries & Filtering** (25 points)

Enhance `GET /api/books` to support filtering and search.

**Implement query parameter support:**

- `?genre=fiction` - Filter by genre
- `?available=true` - Filter by availability (true = available only, false = borrowed only)
- `?search=term` - Search by title (case-insensitive, partial match)
- Support **combining multiple filters** in one request
- Sort all results by `publishedYear` descending (newest first)

---

### **Question 3: Update, Delete & Validation** (30 points)

**Implement these endpoints:**

- `PUT /api/books/:id` - Update book (don't allow changing `authorId`), return 200 or 404
- `PUT /api/authors/:id` - Update author, return 200 or 404
- `DELETE /api/books/:id` - Delete book, return 200 or 404
- `DELETE /api/authors/:id` - Delete author
  - **Must check:** If author has any books, return 400 error
  - If no books, proceed with deletion

---

### **Question 4: Complex Operations & Statistics** (20 points)

**Implement these endpoints:**

- `POST /api/books/:id/borrow` - Accepts `{ "borrowedBy": "Name" }`
  - Validate: Book exists (404) and is available (400 if not)
  - Update: Set `isAvailable: false` and `borrowedBy` field
  - Return updated book (200)

- `POST /api/books/:id/return` - No request body needed
  - Update: Set `isAvailable: true` and `borrowedBy: null`
  - Return updated book (200 or 404)

- `GET /api/stats` - Return statistics object:
  - `totalBooks`, `totalAuthors`, `availableBooks`, `borrowedBooks`
  - `booksByGenre` - object with count for each genre

---

### **Bonus Question: Middleware** (10 points)

**Implement:**

- Request logger middleware - Log `[TIMESTAMP] METHOD URL` for every request
- 404 handler middleware - Handle undefined routes (must be last)
- Validation middleware for `POST /api/books` - Check title length, valid genre, year range (manual validation, not Mongoose)

---

## 📊 Grading Breakdown

| Question | Points | Topics Covered |
|----------|--------|----------------|
| Q1 | 25 | Setup, Schemas, Basic CRUD |
| Q2 | 25 | Query Filters, Search, Sorting |
| Q3 | 30 | Update, Delete, Cascade Logic |
| Q4 | 20 | Complex Updates, Statistics |
| Bonus | 10 | Middleware, Validation |
| **Total** | **110** | **(100 base + 10 bonus)** |

---

## 📚 Allowed Resources

- [Express.js Documentation](https://expressjs.com/)
- [Mongoose Documentation](https://mongoosejs.com/)
- Your course notes and cheatsheet
- **NO AI tools or external code sources**

---

## 🎯 Final Checklist

- [ ] Server runs on port 3000
- [ ] MongoDB connected
- [ ] Both models defined
- [ ] All required endpoints implemented
- [ ] Proper error handling with try-catch
- [ ] Correct HTTP status codes

---

**Good luck! 🚀**
