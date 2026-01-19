# React Lab 03 - Workshop Registration Control Room

## Overview

This project implements a Workshop Registration Control Room focusing on **Chapter 08 (Conditional Rendering)** and **Chapter 09 (Forms & Validation)**. Students can filter workshops, select sessions, and register with a fully controlled form.

## Setup Instructions

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Run Development Server**

   ```bash
   npm run dev
   ```

3. **Build for Production**

   ```bash
   npm run build
   ```

## Features Implemented

### Part 1 - Data Flow & Selection State (Chapter 08)

**State Variables:**

- `workshops` - Array of workshop objects initialized with `INITIAL_WORKSHOPS`
- `filters` - Object with `{ showOnlyOpen, mode, focus }`
- `selectedWorkshopId` - ID of the currently selected workshop
- `lastRegistration` - Most recent registration submission

**Derived Values:**

- `filteredWorkshops` - Workshops filtered based on active filters
- `openCount` - Count of workshops with available seats
- `isFull` - Boolean indicating if selected workshop is full
- `almostFull` - Boolean for workshops with ≤3 seats remaining
- `noResults` - Boolean when no workshops match filters

**Auto-Fallback Logic:**
Uses `useEffect` to automatically select the first available workshop when the currently selected one is filtered out.

### Part 2 - Filter Panel & Workshop Cards (Chapter 08)

**FilterPanel Component:**

- Controlled checkbox for "Show only open seats"
- Segmented buttons for delivery mode (all/in-person/remote)
- Select dropdown for focus area (all/events/forms/assessment)
- Dynamic filter caption showing active filters
- All controls fully controlled via props

**WorkshopCard Component:**

- Conditional class names for selected/full states
- Dynamic seat status badges (Open/Filling fast/Waitlist only)
- Mode and focus badges
- Different icons for in-person (📍) vs remote (🌐) sessions
- Displays tags, instructor, session time, and duration
- Click handler to select workshop

### Part 3 - Registration Form (Chapter 09)

**Form State:**

```javascript
{
  fullName: '',
  email: '',
  experience: 'beginner',
  attendanceType: 'in-person' or 'remote',
  needsEquipment: false,
  dietary: 'none',
  notes: '',
  equipmentDetails: ''
}
```

**Validation:**

- Name must be at least 3 characters
- Email must contain '@' symbol
- Inline error messages displayed below inputs

**Conditional Fields:**

- Attendance type radio - in-person disabled when workshop is remote-only
- Equipment details textarea - only shown when `needsEquipment` is checked
- Dietary helper note - shown when dietary restriction is selected

**Form Submission:**

- Validates all fields before submission
- Calls `onSubmit` with form data + workshopId
- Resets form after successful submission

### Part 4 - Confirmation Panel & Seat Updates

**Seat Management:**

- Increments `seats.taken` when seats are available
- Sets `waitlist: true` flag when workshop is full
- Updates workshop state immutably

**SummaryPanel Component:**

- Renders nothing when `registration` is null
- Displays attendee information
- Shows workshop details and session time
- Status badge (Confirmed/Waitlist) with appropriate styling
- Conditionally shows equipment and dietary notes
- Clear button to reset and allow new registration
- Timestamp of submission

## Components

### App.jsx

Main application component managing:

- Global state (workshops, filters, selection, registration)
- Filter logic and derived values
- Event handlers for filter changes and selection
- Registration submission with seat updates
- Layout and component composition

### FilterPanel.jsx

Filter controls component:

- Checkbox for open seats filter
- Mode selection buttons (all/in-person/remote)
- Focus area dropdown
- Helper text for active filters

### WorkshopCard.jsx

Workshop display card:

- Workshop information display
- Conditional styling based on selection and availability
- Seat status badges
- Tags and metadata
- Selection button

### RegistrationForm.jsx

Registration form with validation:

- Controlled inputs for all fields
- Real-time validation
- Conditional field rendering
- Form submission handling
- Reset after submission

### SummaryPanel.jsx

Registration confirmation display:

- Attendee information summary
- Workshop details
- Status indicator (confirmed/waitlist)
- Optional notes (equipment/dietary)
- Clear functionality

## Conditional Rendering Patterns Used

1. **Ternary operators** - For seat status badges and status text
2. **Logical && operator** - For conditional pills and helper text
3. **Early return** - SummaryPanel returns null when no registration
4. **Conditional classes** - Dynamic className composition
5. **Array.filter()** - Workshop filtering logic
6. **Map with conditional content** - Workshop grid and tags
7. **Disabled attributes** - Form controls based on workshop mode

## Form Validation Patterns

1. **Controlled inputs** - All form fields managed via state
2. **Inline validation** - Real-time error checking on change
3. **Submit validation** - Complete validation before submission
4. **Error state management** - Separate errors object
5. **Conditional field rendering** - Based on form values

## Key Features

✅ Dynamic filtering with multiple criteria
✅ Automatic workshop selection fallback
✅ Conditional rendering throughout UI
✅ Fully controlled form inputs
✅ Real-time validation with error messages
✅ Seat count updates on registration
✅ Waitlist handling for full workshops
✅ Registration confirmation panel
✅ Responsive design with CSS grid
✅ Arabic time format preservation (ق.ظ./ب.ظ.)

## Technologies Used

- React 18
- Vite
- CSS Custom Properties
- ES6+ JavaScript

## Testing Recommendations

- Test all filter combinations
- Try registering for open workshops
- Try registering for full workshops (waitlist)
- Test form validation (invalid email, short name)
- Test conditional fields (equipment, dietary)
- Verify seat count increments correctly
- Test workshop selection fallback when filtering
