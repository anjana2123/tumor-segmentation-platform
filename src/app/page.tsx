'use client';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import MethodSelector from '@/components/MethodSelector';
import ResultsDisplay from '@/components/ResultsDisplay';
import MethodDirectory from '@/components/MethodDirectory';
import { getMethodById } from '@/data/segmentationMethods';

interface BratsCase {
  id: string;
  path: string;
  tumorPercentage: number;
  numSlices: number;
  bestSlice: number;
}

export default function Home() {
  const [bratsCases, setBratsCases] = useState<BratsCase[]>([]);
  const [selectedCase, setSelectedCase] = useState<string>('');
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentMethod, setCurrentMethod] = useState<string>('');
  const [results, setResults] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'segment' | 'directory'>('segment');
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [groundTruth, setGroundTruth] = useState<string | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/cases`)
      .then(res => res.json())
      .then(data => {
        console.log('Loaded cases:', data.cases.length);
        setBratsCases(data.cases);
      })
      .catch(err => console.error('Error loading cases:', err));
  }, []);

  const handleMethodsChange = (methods: string[]) => {
    setSelectedMethods(methods);
  };

  const handleCaseSelect = async (caseId: string) => {
    setSelectedCase(caseId);
    setResults(null);
    
    if (!caseId) {
      setOriginalImage(null);
      setGroundTruth(null);
      return;
    }
    
    setIsLoadingPreview(true);
    try {
      const formData = new FormData();
      formData.append('case_id', caseId);
      
      const response = await fetch(`${API_URL}/preview-case`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setOriginalImage(data.originalImage);
        setGroundTruth(data.groundTruth);
      } else {
        console.error('Preview failed:', response.status);
      }
    } catch (error) {
      console.error('Preview error:', error);
    } finally {
      setIsLoadingPreview(false);
    }
  };

  const handleSegment = async () => {
  if (!selectedCase || selectedMethods.length === 0) {
    alert('Please select a BraTS case and at least one method');
    return;
  }

  setIsProcessing(true);
  setResults(null);
  const allResults: any[] = [];
  
  try {
    // Process each method one by one
    for (let i = 0; i < selectedMethods.length; i++) {
      const methodId = selectedMethods[i];
      const methodName = getMethodById(methodId)?.name || methodId;
      
      setCurrentMethod(`Processing ${i + 1}/${selectedMethods.length}: ${methodName}`);
      
      const formData = new FormData();
      formData.append('case_id', selectedCase);
      formData.append('methods', methodId); // Send only one method
      
      const apiResponse = await fetch(`${API_URL}/segment-brats`, {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        const errorData = await apiResponse.json().catch(() => ({}));
        console.error(`Error with ${methodName}:`, errorData);
        continue; // Skip this method and continue with others
      }
      
      const data = await apiResponse.json();
      
      // Add results from this method
      if (data.results && data.results.length > 0) {
        allResults.push(...data.results);
      }
      
      // Update images on first successful response
      if (i === 0) {
        setOriginalImage(data.originalImage);
        setGroundTruth(data.groundTruth);
      }
    }
    
    // Set all results at the end
    if (allResults.length > 0) {
      setResults(allResults);
    } else {
      alert('No results were generated. Check console for errors.');
    }
    
  } catch (error) {
    console.error('Segmentation error:', error);
    alert(`Error: ${error}\n\nMake sure backend is running.`);
  } finally {
    setIsProcessing(false);
    setCurrentMethod('');
  }
};

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-grow">
        <div className="mb-8 border-b border-gray-200">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('segment')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'segment'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Segmentation Tool
            </button>
            <button
              onClick={() => setActiveTab('directory')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'directory'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Method Directory
            </button>
          </nav>
        </div>

        {activeTab === 'segment' ? (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Select BraTS Case</h3>
                  {bratsCases.length === 0 ? (
                    <p className="text-gray-500">Loading cases...</p>
                  ) : (
                    <div>
                      <select
                        value={selectedCase}
                        onChange={(e) => handleCaseSelect(e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Choose a case...</option>
                        {bratsCases.map(case_ => (
                          <option key={case_.id} value={case_.id}>
                            {case_.id} - Tumor: {case_.tumorPercentage.toFixed(1)}%
                          </option>
                        ))}
                      </select>
                      <p className="mt-2 text-sm text-gray-600">
                        {bratsCases.length} cases available from BraTS 2020 dataset
                      </p>
                    </div>
                  )}
                </div>

                <MethodSelector
                  selectedMethods={selectedMethods}
                  onChange={handleMethodsChange}
                />
                
                <button
                  onClick={handleSegment}
                  disabled={!selectedCase || selectedMethods.length === 0 || isProcessing}
                  className="w-full btn-primary py-3 text-base"
                >
                  {isProcessing ? 'Processing...' : 'Run Segmentation'}
                </button>

                {isProcessing && currentMethod && (
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-blue-900">
                          {currentMethod}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Image Preview</h3>
                {isLoadingPreview ? (
                  <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
                    <p className="text-gray-500">Loading preview...</p>
                  </div>
                ) : originalImage ? (
                  <div className="space-y-4">
                    <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                      <img
                        src={originalImage}
                        alt="BraTS MRI scan"
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs">
                        Original MRI (FLAIR)
                      </div>
                    </div>
                    {groundTruth && (
                      <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                        <img
                          src={groundTruth}
                          alt="Ground truth annotation"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs">
                          Ground Truth (Radiologist Annotation)
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
                    <p className="text-gray-500">Select a case to preview</p>
                  </div>
                )}
                
                {selectedMethods.length > 0 && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-700 font-medium mb-2">
                      Selected Methods ({selectedMethods.length}):
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {selectedMethods.map(methodId => (
                        <span
                          key={methodId}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                        >
                          {methodId}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {results && (
              <>
                <ResultsDisplay
                  results={results}
                  originalImage={originalImage}
                />
                
                <div className="card bg-yellow-50 border border-yellow-200">
                  <h3 className="text-lg font-semibold mb-4 text-yellow-900">Why Some Models May Show Poor Performance</h3>
                  <div className="space-y-3 text-sm text-yellow-800">
                    <p>
                      <strong>Classical and Machine Learning Methods:</strong> These methods (K-Means, Random Forest, Otsu, etc.) work directly on image features and can achieve excellent results without training. K-Means clustering often performs best by identifying intensity-based patterns.
                    </p>
                    <p>
                      <strong>Deep Learning Models Without Medical Pre-training:</strong> Models like UNETR and SegResNet show poor performance because they lack pre-trained weights specifically for medical imaging. These architectures need extensive training on medical datasets to learn tumor patterns.
                    </p>
                    <p>
                      <strong>SAM and MedSAM:</strong> These foundation models are pre-trained on diverse images and can segment objects zero-shot, but may not perfectly understand tumor boundaries without medical-specific fine-tuning.
                    </p>
                    <p>
                      <strong>Research Implications:</strong> This comparison demonstrates that classical methods can be highly competitive with deep learning, especially when computational resources for training are limited. For production systems, deep learning models would require proper training on large medical imaging datasets.
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        ) : (
          <MethodDirectory />
        )}
      </main>

      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-sm text-gray-300 mb-2">
              This is a research platform demonstrating various tumor segmentation approaches.
            </p>
            <p className="text-sm text-gray-300">
              For access to the complete codebase, training scripts, and model weights, please contact: 
              <a href="mailto:anjanaram03@gmail.com" className="text-blue-400 hover:text-blue-300 ml-1">
                anjanaram03@gmail.com
              </a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}