/*
Student ID: [Your Student ID]
Full Name: [Your Full Name]
Date: January 18, 2026

Library Management System API - Complete Solution
*/

const express = require('express');
const mongoose = require('mongoose');

const app = express();

// ============================================
// MIDDLEWARE (BONUS QUESTION)
// ============================================

// Built-in middleware - Parse JSON request bodies
app.use(express.json());

// BONUS: Request logger middleware
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.url}`);
  next();
});

// ============================================
// MONGODB CONNECTION
// ============================================

async function connectDB() {
  try {
    await mongoose.connect('mongodb://localhost:27017/library_db');
    console.log('✅ Connected to MongoDB');
  } catch (error) {
    console.error('❌ MongoDB connection error:', error.message);
    process.exit(1);
  }
}

connectDB();

// ============================================
// MONGOOSE SCHEMAS & MODELS (Q1)
// ============================================

// Author Schema
const authorSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  country: {
    type: String,
    default: 'Unknown'
  },
  booksWritten: {
    type: Number,
    default: 0
  }
}, { 
  timestamps: true 
});

const Author = mongoose.model('Author', authorSchema);

// Book Schema
const bookSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    minlength: 3
  },
  authorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Author',
    required: true
  },
  genre: {
    type: String,
    required: true,
    enum: ['fiction', 'non-fiction', 'science', 'biography', 'history']
  },
  publishedYear: {
    type: Number,
    required: true,
    min: 1900,
    max: new Date().getFullYear()
  },
  pages: {
    type: Number,
    required: true,
    min: 1
  },
  isAvailable: {
    type: Boolean,
    default: true
  },
  borrowedBy: {
    type: String,
    default: null
  }
}, { 
  timestamps: true 
});

const Book = mongoose.model('Book', bookSchema);

// ============================================
// BONUS: VALIDATION MIDDLEWARE
// ============================================

const validateBook = (req, res, next) => {
  const { title, genre, publishedYear } = req.body;
  const validGenres = ['fiction', 'non-fiction', 'science', 'biography', 'history'];
  const currentYear = new Date().getFullYear();

  // Validate title
  if (!title || title.length < 3) {
    return res.status(400).json({ 
      error: 'Title is required and must be at least 3 characters' 
    });
  }

  // Validate genre
  if (!genre || !validGenres.includes(genre)) {
    return res.status(400).json({ 
      error: `Genre must be one of: ${validGenres.join(', ')}` 
    });
  }

  // Validate publishedYear
  if (!publishedYear || publishedYear < 1900 || publishedYear > currentYear) {
    return res.status(400).json({ 
      error: `Published year must be between 1900 and ${currentYear}` 
    });
  }

  next();
};

// ============================================
// AUTHOR ENDPOINTS (Q1)
// ============================================

// Create new author
app.post('/api/authors', async (req, res) => {
  try {
    const author = await Author.create(req.body);
    res.status(201).json(author);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Get all authors
app.get('/api/authors', async (req, res) => {
  try {
    const authors = await Author.find();
    res.status(200).json(authors);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get author by ID
app.get('/api/authors/:id', async (req, res) => {
  try {
    const author = await Author.findById(req.params.id);
    
    if (!author) {
      return res.status(404).json({ error: 'Author not found' });
    }
    
    res.status(200).json(author);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update author (Q3)
app.put('/api/authors/:id', async (req, res) => {
  try {
    const author = await Author.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );
    
    if (!author) {
      return res.status(404).json({ error: 'Author not found' });
    }
    
    res.status(200).json(author);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Delete author with cascade check (Q3)
app.delete('/api/authors/:id', async (req, res) => {
  try {
    // Check if author has books
    const bookCount = await Book.countDocuments({ authorId: req.params.id });
    
    if (bookCount > 0) {
      return res.status(400).json({ 
        error: 'Cannot delete author with existing books' 
      });
    }
    
    const author = await Author.findByIdAndDelete(req.params.id);
    
    if (!author) {
      return res.status(404).json({ error: 'Author not found' });
    }
    
    res.status(200).json({ message: 'Author deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================
// BOOK ENDPOINTS (Q1, Q2, Q3, Q4)
// ============================================

// Create new book (Q1 + BONUS validation middleware)
app.post('/api/books', validateBook, async (req, res) => {
  try {
    // Verify authorId exists
    const authorExists = await Author.findById(req.body.authorId);
    
    if (!authorExists) {
      return res.status(400).json({ error: 'Author not found' });
    }
    
    const book = await Book.create(req.body);
    res.status(201).json(book);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Get all books with filtering and search (Q1 + Q2)
app.get('/api/books', async (req, res) => {
  try {
    let filter = {};
    
    // Q2: Filter by genre
    if (req.query.genre) {
      filter.genre = req.query.genre;
    }
    
    // Q2: Filter by availability
    if (req.query.available !== undefined) {
      filter.isAvailable = req.query.available === 'true';
    }
    
    // Q2: Search by title (case-insensitive, partial match)
    if (req.query.search) {
      filter.title = { $regex: req.query.search, $options: 'i' };
    }
    
    // Q2: Sort by publishedYear descending
    const books = await Book.find(filter).sort({ publishedYear: -1 });
    res.status(200).json(books);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update book (Q3)
app.put('/api/books/:id', async (req, res) => {
  try {
    // Don't allow updating authorId
    const { authorId, ...updateData } = req.body;
    
    const book = await Book.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true, runValidators: true }
    );
    
    if (!book) {
      return res.status(404).json({ error: 'Book not found' });
    }
    
    res.status(200).json(book);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Delete book (Q3)
app.delete('/api/books/:id', async (req, res) => {
  try {
    const book = await Book.findByIdAndDelete(req.params.id);
    
    if (!book) {
      return res.status(404).json({ error: 'Book not found' });
    }
    
    res.status(200).json({ message: 'Book deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Borrow book (Q4)
app.post('/api/books/:id/borrow', async (req, res) => {
  try {
    const { borrowedBy } = req.body;
    
    const book = await Book.findById(req.params.id);
    
    if (!book) {
      return res.status(404).json({ error: 'Book not found' });
    }
    
    if (!book.isAvailable) {
      return res.status(400).json({ error: 'Book is already borrowed' });
    }
    
    book.isAvailable = false;
    book.borrowedBy = borrowedBy;
    await book.save();
    
    res.status(200).json(book);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Return book (Q4)
app.post('/api/books/:id/return', async (req, res) => {
  try {
    const book = await Book.findById(req.params.id);
    
    if (!book) {
      return res.status(404).json({ error: 'Book not found' });
    }
    
    book.isAvailable = true;
    book.borrowedBy = null;
    await book.save();
    
    res.status(200).json(book);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================
// STATISTICS ENDPOINT (Q4)
// ============================================

app.get('/api/stats', async (req, res) => {
  try {
    // Total counts
    const totalBooks = await Book.countDocuments();
    const totalAuthors = await Author.countDocuments();
    const availableBooks = await Book.countDocuments({ isAvailable: true });
    const borrowedBooks = await Book.countDocuments({ isAvailable: false });
    
    // Count books by genre
    const genreCounts = await Book.aggregate([
      {
        $group: {
          _id: '$genre',
          count: { $sum: 1 }
        }
      }
    ]);
    
    // Format genre counts
    const booksByGenre = {
      fiction: 0,
      'non-fiction': 0,
      science: 0,
      biography: 0,
      history: 0
    };
    
    genreCounts.forEach(item => {
      booksByGenre[item._id] = item.count;
    });
    
    res.status(200).json({
      totalBooks,
      totalAuthors,
      availableBooks,
      borrowedBooks,
      booksByGenre
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================
// BONUS: 404 HANDLER (MUST BE LAST!)
// ============================================

app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// ============================================
// START SERVER
// ============================================

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});