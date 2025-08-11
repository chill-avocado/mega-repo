import React from 'react';
import FormFieldDefinition from '../common/FormFieldDefinition';

/**
 * Advanced Settings section of the agent form
 * 
 * @param {Object} props Component props
 * @param {Object} props.formData Current form data values
 * @param {Function} props.handleInputChange Handler for input changes
 * @param {Object} props.metadata Field metadata from the backend
 */
const AdvancedSettingsSection = ({ formData, handleInputChange, metadata }) => {
  // Get fields from metadata
  const fields = metadata?.AdvancedSettingsSection || [];

  // Handle field value changes
  const handleFieldChange = (name, value) => {
    const field = fields.find(f => f.name === name);
    if (field && field.type === 'checkbox') {
      handleInputChange({
        target: {
          name,
          type: 'checkbox',
          checked: value === 'true'
        }
      });
    } else {
      handleInputChange({
        target: {
          name,
          value
        }
      });
    }
  };

  return (
    <div id="advanced-section">
      <h3 className="section-title">Advanced Settings</h3>
      
      <FormFieldDefinition
        fields={fields}
        values={formData}
        onChange={handleFieldChange}
        idPrefix="advanced_"
      />
    </div>
  );
};

export default AdvancedSettingsSection;
