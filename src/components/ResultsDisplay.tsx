'use client';

import React, { useState } from 'react';
import { getMethodById } from '@/data/segmentationMethods';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Clock, TrendingUp } from 'lucide-react';

interface ResultMetrics {
  diceScore: number;
  iou: number;
  processingTime: number;
  precision: number;
  recall: number;
}

interface SegmentationResult {
  methodId: string;
  segmentedImage: string;
  metrics: ResultMetrics;
}

interface ResultsDisplayProps {
  results: SegmentationResult[];
  originalImage: string | null;
}

export default function ResultsDisplay({ results, originalImage }: ResultsDisplayProps) {
  const [selectedResult, setSelectedResult] = useState<string | null>(results[0]?.methodId || null);
  const [comparisonMode, setComparisonMode] = useState<'single' | 'grid'>('single');

  const selectedResultData = results.find(r => r.methodId === selectedResult);
  const selectedMethod = selectedResultData ? getMethodById(selectedResultData.methodId) : null;

  // Prepare data for metrics chart
  const metricsChartData = results.map(result => {
    const method = getMethodById(result.methodId);
    return {
      name: method?.name.substring(0, 15) || result.methodId,
      'Dice Score': (result.metrics.diceScore * 100).toFixed(1),
      'IoU': (result.metrics.iou * 100).toFixed(1),
      'Precision': (result.metrics.precision * 100).toFixed(1),
      'Recall': (result.metrics.recall * 100).toFixed(1),
    };
  });

  const performanceChartData = results.map(result => {
    const method = getMethodById(result.methodId);
    return {
      name: method?.name.substring(0, 15) || result.methodId,
      'Time (s)': result.metrics.processingTime.toFixed(2),
    };
  }).sort((a, b) => parseFloat(a['Time (s)']) - parseFloat(b['Time (s)']));

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">Segmentation Results</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setComparisonMode('single')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                comparisonMode === 'single'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Single View
            </button>
            <button
              onClick={() => setComparisonMode('grid')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                comparisonMode === 'grid'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Grid Comparison
            </button>
          </div>
        </div>

        {comparisonMode === 'single' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Image Comparison */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-3">Original vs Segmented</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-500 mb-2">Original</p>
                  <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                    {originalImage && (
                      <img src={originalImage} alt="Original" className="w-full h-full object-contain" />
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-2">Segmented</p>
                  <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                    {selectedResultData && (
                      <img src={selectedResultData.segmentedImage} alt="Segmented" className="w-full h-full object-contain" />
                    )}
                  </div>
                </div>
              </div>

              {/* Method Selector Tabs */}
              <div className="mt-4 flex flex-wrap gap-2">
                {results.map(result => {
                  const method = getMethodById(result.methodId);
                  return (
                    <button
                      key={result.methodId}
                      onClick={() => setSelectedResult(result.methodId)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                        selectedResult === result.methodId
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      {method?.name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Metrics for Selected Method */}
            {selectedResultData && selectedMethod && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">
                  {selectedMethod.name} Performance
                </h3>
                
                <div className="space-y-3">
                  <MetricCard
                    label="Dice Coefficient"
                    value={(selectedResultData.metrics.diceScore * 100).toFixed(2)}
                    unit="%"
                    description="Overlap between prediction and ground truth"
                  />
                  <MetricCard
                    label="IoU (Jaccard Index)"
                    value={(selectedResultData.metrics.iou * 100).toFixed(2)}
                    unit="%"
                    description="Intersection over union metric"
                  />
                  <MetricCard
                    label="Precision"
                    value={(selectedResultData.metrics.precision * 100).toFixed(2)}
                    unit="%"
                    description="True positives / (True positives + False positives)"
                  />
                  <MetricCard
                    label="Recall (Sensitivity)"
                    value={(selectedResultData.metrics.recall * 100).toFixed(2)}
                    unit="%"
                    description="True positives / (True positives + False negatives)"
                  />
                  <div className="flex items-center space-x-2 p-3 bg-gray-50 rounded-lg">
                    <Clock className="h-5 w-5 text-gray-500" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">
                        {selectedResultData.metrics.processingTime.toFixed(2)}s
                      </p>
                      <p className="text-xs text-gray-500">Processing time</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {results.map(result => {
              const method = getMethodById(result.methodId);
              return (
                <div key={result.methodId} className="border border-gray-200 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-gray-900 mb-2 truncate">
                    {method?.name}
                  </h4>
                  <div className="aspect-square bg-gray-100 rounded overflow-hidden mb-2">
                    <img src={result.segmentedImage} alt={method?.name} className="w-full h-full object-contain" />
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Dice:</span>
                      <span className="font-medium">{(result.metrics.diceScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Time:</span>
                      <span className="font-medium">{result.metrics.processingTime.toFixed(2)}s</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Performance Comparison Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Accuracy Metrics Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={80} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Bar dataKey="Dice Score" fill="#0ea5e9" />
              <Bar dataKey="IoU" fill="#8b5cf6" />
              <Bar dataKey="Precision" fill="#10b981" />
              <Bar dataKey="Recall" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Processing Time Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={performanceChartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="Time (s)" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Best Performer Summary */}
      <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <div className="flex items-start space-x-3">
          <TrendingUp className="h-6 w-6 text-blue-600 mt-1" />
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Performance Summary</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Highest Dice Score</p>
                <p className="font-semibold text-gray-900">
                  {getMethodById(
                    [...results].sort((a, b) => b.metrics.diceScore - a.metrics.diceScore)[0].methodId
                  )?.name}
                </p>
                <p className="text-xs text-blue-600">
                  {(Math.max(...results.map(r => r.metrics.diceScore)) * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-gray-600">Fastest Processing</p>
                <p className="font-semibold text-gray-900">
                  {getMethodById(
                    [...results].sort((a, b) => a.metrics.processingTime - b.metrics.processingTime)[0].methodId
                  )?.name}
                </p>
                <p className="text-xs text-blue-600">
                  {Math.min(...results.map(r => r.metrics.processingTime)).toFixed(2)}s
                </p>
              </div>
              <div>
                <p className="text-gray-600">Best Precision</p>
                <p className="font-semibold text-gray-900">
                  {getMethodById(
                    [...results].sort((a, b) => b.metrics.precision - a.metrics.precision)[0].methodId
                  )?.name}
                </p>
                <p className="text-xs text-blue-600">
                  {(Math.max(...results.map(r => r.metrics.precision)) * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit, description }: { label: string; value: string; unit: string; description: string }) {
  return (
    <div className="p-3 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between mb-1">
        <p className="text-sm font-medium text-gray-900">{label}</p>
        <p className="text-lg font-semibold text-blue-600">
          {value}<span className="text-sm">{unit}</span>
        </p>
      </div>
      <p className="text-xs text-gray-600">{description}</p>
    </div>
  );
}