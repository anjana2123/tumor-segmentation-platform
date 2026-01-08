'use client';

import React, { useState } from 'react';
import { segmentationMethods, SegmentationMethod } from '@/data/segmentationMethods';
import { Search, BookOpen, Zap, CheckCircle, XCircle, FileText, Clock, Award } from 'lucide-react';

export default function MethodDirectory() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedMethod, setSelectedMethod] = useState<SegmentationMethod | null>(null);

  const categories = [
    { id: 'all', name: 'All Methods' },
    { id: 'classical', name: 'Classical' },
    { id: 'machine-learning', name: 'Machine Learning' },
    { id: 'deep-learning', name: 'Deep Learning' },
  ];

  const filteredMethods = segmentationMethods.filter(method => {
    const matchesSearch = method.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         method.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || method.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left Sidebar: Method List */}
      <div className="lg:col-span-1">
        <div className="card sticky top-4">
          <h3 className="text-lg font-semibold mb-4">Browse Methods</h3>
          
          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search methods..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
            />
          </div>

          {/* Category Filter */}
          <div className="flex flex-wrap gap-2 mb-4">
            {categories.map(cat => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  selectedCategory === cat.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {cat.name}
              </button>
            ))}
          </div>

          {/* Methods List */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {filteredMethods.map(method => (
              <button
                key={method.id}
                onClick={() => setSelectedMethod(method)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedMethod?.id === method.id
                    ? 'bg-blue-50 border-2 border-blue-500'
                    : 'bg-gray-50 border-2 border-transparent hover:bg-gray-100'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {method.name}
                    </p>
                    <p className="text-xs text-gray-600 mt-0.5">
                      {method.category.replace('-', ' ')}
                    </p>
                  </div>
                  <span className="text-xs text-gray-500 ml-2">
                    {method.referenceYear}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right Content: Method Details */}
      <div className="lg:col-span-2">
        {selectedMethod ? (
          <div className="space-y-6">
            {/* Header */}
            <div className="card">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    {selectedMethod.name}
                  </h2>
                  <div className="flex items-center space-x-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      selectedMethod.category === 'classical' ? 'bg-blue-100 text-blue-800' :
                      selectedMethod.category === 'machine-learning' ? 'bg-purple-100 text-purple-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {selectedMethod.category.replace('-', ' ').toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-600">
                      Reference Year: {selectedMethod.referenceYear}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-gray-700 leading-relaxed">
                {selectedMethod.description}
              </p>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="card bg-blue-50 border-blue-200">
                <div className="flex items-center space-x-2 mb-2">
                  <Award className="h-5 w-5 text-blue-600" />
                  <p className="text-sm font-medium text-gray-700">Accuracy</p>
                </div>
                <p className="text-lg font-semibold text-blue-900">
                  {selectedMethod.accuracy}
                </p>
              </div>
              
              <div className="card bg-purple-50 border-purple-200">
                <div className="flex items-center space-x-2 mb-2">
                  <Clock className="h-5 w-5 text-purple-600" />
                  <p className="text-sm font-medium text-gray-700">Complexity</p>
                </div>
                <p className="text-sm font-semibold text-purple-900">
                  {selectedMethod.computationalComplexity}
                </p>
              </div>

              <div className="card bg-green-50 border-green-200 col-span-2 md:col-span-1">
                <div className="flex items-center space-x-2 mb-2">
                  <Zap className="h-5 w-5 text-green-600" />
                  <p className="text-sm font-medium text-gray-700">Category</p>
                </div>
                <p className="text-sm font-semibold text-green-900 capitalize">
                  {selectedMethod.category.replace('-', ' ')}
                </p>
              </div>
            </div>

            {/* Algorithm Details */}
            <div className="card">
              <div className="flex items-center space-x-2 mb-3">
                <BookOpen className="h-5 w-5 text-blue-600" />
                <h3 className="text-lg font-semibold">Algorithm</h3>
              </div>
              <p className="text-gray-700 mb-4">{selectedMethod.algorithm}</p>
              
              <h4 className="text-sm font-semibold text-gray-900 mb-2">Mathematical Logic</h4>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-sm text-gray-700 leading-relaxed font-mono">
                  {selectedMethod.logic}
                </p>
              </div>
            </div>

            {/* Advantages & Limitations */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card bg-green-50 border-green-200">
                <div className="flex items-center space-x-2 mb-3">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Advantages</h3>
                </div>
                <ul className="space-y-2">
                  {selectedMethod.advantages.map((adv, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-green-600 mt-0.5">•</span>
                      <span className="text-sm text-gray-700">{adv}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="card bg-red-50 border-red-200">
                <div className="flex items-center space-x-2 mb-3">
                  <XCircle className="h-5 w-5 text-red-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Limitations</h3>
                </div>
                <ul className="space-y-2">
                  {selectedMethod.limitations.map((lim, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-red-600 mt-0.5">•</span>
                      <span className="text-sm text-gray-700">{lim}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Use Cases */}
            <div className="card">
              <div className="flex items-center space-x-2 mb-3">
                <FileText className="h-5 w-5 text-blue-600" />
                <h3 className="text-lg font-semibold">Clinical Use Cases</h3>
              </div>
              <div className="grid grid-cols-1 gap-3">
                {selectedMethod.useCases.map((useCase, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-sm text-gray-700">{useCase}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Key Papers */}
            <div className="card bg-gray-50">
              <h3 className="text-lg font-semibold mb-3">Key Research Papers</h3>
              <div className="space-y-2">
                {selectedMethod.keyPapers.map((paper, idx) => (
                  <div key={idx} className="p-3 bg-white rounded border border-gray-200">
                    <p className="text-sm text-gray-800 font-medium">{paper}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="card h-full flex items-center justify-center">
            <div className="text-center">
              <BookOpen className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Select a Method
              </h3>
              <p className="text-gray-600">
                Choose a segmentation method from the list to view its details
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}