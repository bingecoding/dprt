/***************************************************************************
 * Copyright (c) 2014 Michael Walch                                        *
 *                                                                         *
 *   This project is based on LuxRays ; see https://luxcorerender.org       *
 *   LuxRays is the part of LuxRender dedicated to accelerate the          *
 *   ray intersection process by using GPUs.                               *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H

#include <iostream>
#include <cmath>

#include "lighttree/lighttree.h"

struct ClusterErrorBound {
    
    bool operator()(const Cluster *h1, const Cluster *h2) const {
        return h1->errorBound.Y() < h2->errorBound.Y();
    }
};

template <typename T, typename F>
class PriorityQueue {
    
public:
    
    PriorityQueue(unsigned int size)
    : m_arity(2), m_size(size), m_functor()
    {
        m_A = new T[m_size];
        m_currIdx = 0;
    }
    
    bool Empty()
    {
        if(m_currIdx <= 0){
            return true;
        }
        
        return false;
    }
    
    T Top()
    {
        if(m_size < 0) {
            return NULL;
        }
        
        T front = m_A[0];
        return front;
    }
    
    void Pop()
    {
        if(m_size <= 0 || m_currIdx <= 0) {
            return;
        }
        
        m_currIdx--;
        m_A[0] = m_A[m_currIdx];
        if(m_currIdx <= 0) {
            return;
        }

        PercolateDown(0);
    }
    
    void Push(T element)
    {
        if(m_size < 0 || m_currIdx >= m_size) {
            return;
        }
        
        Insert(element);
        m_currIdx++;
        
        return;
    }
    
    void Insert(T element) {
        
        int index = m_currIdx;
        for(; index > 0 && element->diff > m_A[index/2]->diff; index /=2)
            m_A[index] = m_A[index/2];
        m_A[index] = element;
        
    }
    
    void PercolateDown(int index)
    {
        int child;
        T tmp = m_A[index];
        
        for(; index*2 <= m_currIdx; index = child) {
            child = index*2;
            if(child != m_currIdx && m_A[child+1]->diff > m_A[child]->diff)
                child++;
            if(m_A[child]->diff > tmp->diff)
                m_A[index] = m_A[child];
            else
                break;
        }
        
        m_A[index] = tmp;
    }
    
    void Print(){
    
        for(int i=0; i < m_currIdx; i++) {
            std::cout << m_A[i]->errorBound.Y() << std::endl;
        }
        
    }
    
private:
    
    unsigned int Sibling(int index, int j)
    {
        return m_arity*index + j;
    }
    unsigned int Parent(int index)
    {
        float d = m_arity;
        float parent = std::floor((index-1) / d);
        return parent;
    }
    
    unsigned int m_arity;
    unsigned int m_size;
    unsigned int m_currIdx;
    
    F m_functor;
    T *m_A;
    
};

#endif
