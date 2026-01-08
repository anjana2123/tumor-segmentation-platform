'use client';

import React, { useState } from 'react';
import { segmentationMethods, getCategoryMethods } from '@/data/segmentationMethods';
import { ChevronDown, ChevronRight, Info } from 'lucide-react';

interface MethodSelectorProps {
  selectedMethods: string[];
  onChange: (methods: string[]) => void;
}

export default function MethodSelector({ selectedMethods, onChange }: MethodSelectorProps) {
  const [expandedCategories, setExpandedCategories] = useState<string[]>([
    'classical', 
    'machine-learning', 
    'deep-learning'
  ]);
  const [hoveredMethod, setHoveredMethod] = useState<string | null>(null);

  const categories = [
    { id: 'classical', name: 'Classical Methods', color: 'blue' },
    { id: 'machine-learning', name: 'Machine Learning', color: 'purple' },
    { id: 'deep-learning', name: 'Deep Learning', color: 'green' },
  ];

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories(prev =>
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const toggleMethod = (methodId: string) => {
    onChange(
      selectedMethods.includes(methodId)
        ? selectedMethods.filter(id => id !== methodId)
        : [...selectedMethods, methodId]
    );
  };

  const selectAll = (categoryId: string) => {
    const categoryMethods = getCategoryMethods(categoryId);
    const categoryMethodIds = categoryMethods.map(m => m.id);
    const allSelected = categoryMethodIds.every(id => selectedMethods.includes(id));
    
    if (allSelected) {
      onChange(selectedMethods.filter(id => !categoryMethodIds.includes(id)));
    } else {
      const newSelection = [...new Set([...selectedMethods, ...categoryMethodIds])];
      onChange(newSelection);
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Select Segmentation Methods</h3>
        <span className="text-sm text-gray-600">
          {selectedMethods.length} selected
        </span>
      </div>

      <div className="space-y-3">
        {categories.map(category => {
          const methods = getCategoryMethods(category.id);
          const isExpanded = expandedCategories.includes(category.id);
          const categoryMethodIds = methods.map(m => m.id);
          const allSelected = categoryMethodIds.every(id => selectedMethods.includes(id));

          return (
            <div key={category.id} className="border border-gray-200 rounded-lg overflow-hidden">
              <div className="bg-gray-50 px-4 py-3">
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => toggleCategory(category.id)}
                    className="flex items-center space-x-2 flex-1 text-left"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-gray-500" />
                    )}
                    <span className="font-medium text-gray-900">{category.name}</span>
                    <span className={`
                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                      ${category.color === 'blue' ? 'bg-blue-100 text-blue-800' : ''}
                      ${category.color === 'purple' ? 'bg-purple-100 text-purple-800' : ''}
                      ${category.color === 'green' ? 'bg-green-100 text-green-800' : ''}
                    `}>
                      {methods.length}
                    </span>
                  </button>
                  
                  <button
                    onClick={() => selectAll(category.id)}
                    className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                  >
                    {allSelected ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="p-2 space-y-1">
                  {methods.map(method => (
                    <label
                      key={method.id}
                      onMouseEnter={() => setHoveredMethod(method.id)}
                      onMouseLeave={() => setHoveredMethod(null)}
                      className={`
                        flex items-start space-x-3 p-3 rounded-md cursor-pointer
                        transition-colors duration-150
                        ${selectedMethods.includes(method.id)
                          ? 'bg-blue-50 border border-blue-200'
                          : 'hover:bg-gray-50 border border-transparent'
                        }
                      `}
                    >
                      <input
                        type="checkbox"
                        checked={selectedMethods.includes(method.id)}
                        onChange={() => toggleMethod(method.id)}
                        className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-gray-900">
                            {method.name}
                          </span>
                          {hoveredMethod === method.id && (
                            <Info className="h-3 w-3 text-gray-400" />
                          )}
                        </div>
                        <p className="text-xs text-gray-600 mt-0.5 line-clamp-2">
                          {method.description}
                        </p>
                      </div>
                    </label>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {selectedMethods.length > 0 && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            You've selected {selectedMethods.length} method{selectedMethods.length !== 1 ? 's' : ''}. 
            Results will be compared side-by-side with performance metrics.
          </p>
        </div>
      )}
    </div>
  );
}