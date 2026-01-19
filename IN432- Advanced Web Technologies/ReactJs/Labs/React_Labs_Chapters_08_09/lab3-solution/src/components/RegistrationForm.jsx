import React, { useState } from 'react';

/**
 * TODO (Chapter 09):
 * 1. Accept props: { workshop, onSubmit, disabled }.
 * 2. Use useState to manage a form object, e.g.:
 *    {
 *      fullName: '',
 *      email: '',
 *      experience: 'beginner',
 *      attendanceType: workshop.mode === 'remote' ? 'remote' : 'in-person',
 *      needsEquipment: false,
 *      dietary: 'none',
 *      notes: ''
 *    }
 * 3. Build controlled inputs:
 *    - Text inputs for name/email (with simple validation).
 *    - Select for experience level.
 *    - Radio group for attendanceType (remote vs in-person).
 *    - Checkbox for needsEquipment.
 *    - Textarea for notes.
 * 4. Show inline errors (e.g., email missing '@', name < 3 chars).
 * 5. On submit:
 *    - Prevent default.
 *    - If disabled or invalid, bail out.
 *    - Call onSubmit with { ...formData, workshopId: workshop.id }.
 *    - Reset the form and optionally show a success message.
 * 6. Stretch ideas:
 *    - Disable physical attendance choice when workshop.mode === 'remote'.
 *    - Reveal an extra textarea when needsEquipment === true.
 */
function RegistrationForm({ workshop, onSubmit }) {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    experience: 'beginner',
    attendanceType: workshop.mode === 'remote' ? 'remote' : 'in-person',
    needsEquipment: false,
    dietary: 'none',
    notes: '',
    equipmentDetails: ''
  });

  const [errors, setErrors] = useState({});

  const validateField = (name, value) => {
    switch (name) {
      case 'fullName':
        return value.length < 3 ? 'Name must be at least 3 characters' : '';
      case 'email':
        return !value.includes('@') ? 'Email must contain @' : '';
      default:
        return '';
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    const error = validateField(name, value);
    setErrors(prev => ({ ...prev, [name]: error }));
  };

  const handleCheckbox = (e) => {
    const { name, checked } = e.target;
    setFormData(prev => ({ ...prev, [name]: checked }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validate all fields
    const newErrors = {
      fullName: validateField('fullName', formData.fullName),
      email: validateField('email', formData.email)
    };

    setErrors(newErrors);

    // Check if there are any errors
    const hasErrors = Object.values(newErrors).some(err => err !== '');
    if (hasErrors) return;

    // Submit the form
    onSubmit({ ...formData, workshopId: workshop.id });

    // Reset form
    setFormData({
      fullName: '',
      email: '',
      experience: 'beginner',
      attendanceType: workshop.mode === 'remote' ? 'remote' : 'in-person',
      needsEquipment: false,
      dietary: 'none',
      notes: '',
      equipmentDetails: ''
    });
    setErrors({});
  };

  return (
    <form className="registration-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="fullName">Full Name *</label>
        <input
          type="text"
          id="fullName"
          name="fullName"
          value={formData.fullName}
          onChange={handleChange}
          required
        />
        {errors.fullName && <p className="error-text">{errors.fullName}</p>}
      </div>

      <div className="form-group">
        <label htmlFor="email">Email *</label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          required
        />
        {errors.email && <p className="error-text">{errors.email}</p>}
      </div>

      <div className="form-group">
        <label htmlFor="experience">Experience Level</label>
        <select
          id="experience"
          name="experience"
          value={formData.experience}
          onChange={handleChange}
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">Attendance Type *</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="attendanceType"
              value="in-person"
              checked={formData.attendanceType === 'in-person'}
              onChange={handleChange}
              disabled={workshop.mode === 'remote'}
            />
            <span>In-Person</span>
          </label>
          <label>
            <input
              type="radio"
              name="attendanceType"
              value="remote"
              checked={formData.attendanceType === 'remote'}
              onChange={handleChange}
            />
            <span>Remote</span>
          </label>
        </div>
        {workshop.mode === 'remote' && (
          <p className="form-hint">This workshop is remote-only</p>
        )}
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="needsEquipment"
            checked={formData.needsEquipment}
            onChange={handleCheckbox}
          />
          <span>I need equipment</span>
        </label>
      </div>

      {formData.needsEquipment && (
        <div className="form-group">
          <label htmlFor="equipmentDetails">Equipment Details</label>
          <textarea
            id="equipmentDetails"
            name="equipmentDetails"
            value={formData.equipmentDetails}
            onChange={handleChange}
            placeholder="Please specify what equipment you need..."
            rows="3"
          />
        </div>
      )}

      <div className="form-group">
        <label htmlFor="dietary">Dietary Restrictions</label>
        <select
          id="dietary"
          name="dietary"
          value={formData.dietary}
          onChange={handleChange}
        >
          <option value="none">None</option>
          <option value="vegetarian">Vegetarian</option>
          <option value="vegan">Vegan</option>
          <option value="gluten-free">Gluten-Free</option>
          <option value="other">Other</option>
        </select>
        {formData.dietary !== 'none' && (
          <p className="form-hint">We'll make sure to accommodate your dietary needs</p>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="notes">Additional Notes</label>
        <textarea
          id="notes"
          name="notes"
          value={formData.notes}
          onChange={handleChange}
          placeholder="Any special requirements or questions?"
          rows="4"
        />
      </div>

      <button type="submit" className="submit-btn">
        Register for Workshop
      </button>
    </form>
  );
}

export default RegistrationForm;


