import React from 'react';

export default function Header() {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
              Brain Tumor Segmentation Platform
            </h1>
            <p className="mt-1 text-sm sm:text-base text-gray-600">
              Research v1 - Some models require additional training on medical imaging data
            </p>
          </div>
          
          <div className="hidden sm:flex items-center space-x-4">
            <div className="text-right">
              <p className="text-xs text-gray-500">Developed by</p>
              <p className="text-sm font-medium text-gray-700">Anjana Ramachandran</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}