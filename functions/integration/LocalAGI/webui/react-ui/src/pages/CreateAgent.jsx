import { useState, useEffect } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { agentApi } from '../utils/api';
import AgentForm from '../components/AgentForm';

function CreateAgent() {
  const navigate = useNavigate();
  const { showToast } = useOutletContext();
  const [loading, setLoading] = useState(false);
  const [metadata, setMetadata] = useState(null);
  const [formData, setFormData] = useState({});

  useEffect(() => {
    document.title = 'Create Agent - LocalAGI';
    return () => {
      document.title = 'LocalAGI'; // Reset title when component unmounts
    };
  }, []);

  // Fetch metadata on component mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        // Fetch metadata from the dedicated endpoint
        const response = await agentApi.getAgentConfigMetadata();
        if (response) {
          setMetadata(response);
        }
      } catch (error) {
        console.error('Error fetching metadata:', error);
        // Continue without metadata, the form will use default fields
      }
    };

    fetchMetadata();
  }, []);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      showToast('Agent name is required', 'error');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await agentApi.createAgent(formData);
      showToast(`Agent "${formData.name}" created successfully`, 'success');
      navigate(`/settings/${formData.name}`);
    } catch (err) {
      showToast(`Error creating agent: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="create-agent-container">
      <header className="page-header">
        <h1>
          <i className="fas fa-plus-circle"></i> Create New Agent
        </h1>
      </header>
      
      <div className="create-agent-content">
        <div className="section-box">
          <h2>
            <i className="fas fa-robot"></i> Agent Configuration
          </h2>
          
          <AgentForm
            formData={formData}
            setFormData={setFormData}
            onSubmit={handleSubmit}
            loading={loading}
            submitButtonText="Create Agent"
            isEdit={false}
            metadata={metadata}
          />
        </div>
      </div>
    </div>
  );
}

export default CreateAgent;
