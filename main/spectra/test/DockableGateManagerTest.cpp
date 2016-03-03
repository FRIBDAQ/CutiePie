//    This software is Copyright by the Board of Trustees of Michigan
//    State University (c) Copyright 2016.
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//    Authors:
//    Jeromy Tompkins
//    NSCL
//    Michigan State University
//    East Lansing, MI 48824-1321


#ifndef GGATETEST_H
#define GGATETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Asserter.h>


#define private public
#define protected public
#include "DockableGateManager.h"
#undef protected 
#undef private


#include "SpecTclRESTInterface.h"
#include "MultiSpectrumView.h"
#include "GSlice.h"
#include "GGate.h"
#include "Utility.h"

#include <GateInfo.h>

#include <iostream>

using namespace std;

namespace Viewer
{

class DockableGateManagerTest : public CppUnit::TestFixture
{
  public:
    CPPUNIT_TEST_SUITE( DockableGateManagerTest );
    CPPUNIT_TEST( onGateListChanged_0 );
    CPPUNIT_TEST( onGateListChanged_1 );
    CPPUNIT_TEST_SUITE_END();

  public:
    void setUp();
    void tearDown();

  protected:
    void onGateListChanged_0();
    void onGateListChanged_1();

};

CPPUNIT_TEST_SUITE_REGISTRATION(DockableGateManagerTest);

#endif

void DockableGateManagerTest::setUp() 
{
}

void DockableGateManagerTest::tearDown() 
{
}


void DockableGateManagerTest::onGateListChanged_0()
{
  // ensure that the existing list items do not delete if they are 
  // just updated.
  std::shared_ptr<SpecTclInterface> pSpecTcl(new SpecTclRESTInterface);
  pSpecTcl->enableGatePolling(false);
  auto* pList = pSpecTcl->getGateList();

  // Create two gates
  auto pSlice = make_unique(new GSlice(SpJs::Slice("slice", "x", 0, 1)));
  auto pGate = make_unique(new GGate(SpJs::Band("gate", "x", "y", 
                                                {{0, 1}, {1, 2}, {2, 3}})));

  // Add them to the interface
  pList->addCut1D( move(pSlice) );
  pList->addCut2D( move(pGate) );

  MultiSpectrumView viewer(pSpecTcl);
  DockableGateManager manager(viewer, pSpecTcl);

  // first populate
  manager.onGateListChanged();

  auto itemList0 = manager.getItems();
  CPPUNIT_ASSERT( 2 == itemList0.size() );

  // second populate with same exact set of gates...
  manager.onGateListChanged();

  auto itemList1 = manager.getItems();
  CPPUNIT_ASSERT( 2 == itemList1.size() );

  // Makes sure we have the same items...
  for ( int row=0; row<itemList1.size(); ++row) {
    CPPUNIT_ASSERT( itemList0.at(row) == itemList1.at(row) );
  }

}

void DockableGateManagerTest::onGateListChanged_1()
{
  // ensure that the existing list items do not delete if they are 
  // just updated.
  std::shared_ptr<SpecTclInterface> pSpecTcl(new SpecTclRESTInterface);
  pSpecTcl->enableGatePolling(false);
  auto* pList = pSpecTcl->getGateList();

  // Create two gates
  auto pSlice = make_unique(new GSlice(SpJs::Slice("slice", "x", 0, 1)));
  auto pGate = make_unique(new GGate(SpJs::Band("gate", "x", "y", 
                                                {{0, 1}, {1, 2}, {2, 3}})));
  auto pGate1 = make_unique(new GGate(SpJs::Band("gate1", "x", "y", 
                                                {{0, 1}, {1, 2}, {2, 3}})));

  // Add them to the interface
  pList->addCut1D( move(pSlice) );
  pList->addCut2D( move(pGate) );

  MultiSpectrumView viewer(pSpecTcl);
  DockableGateManager manager(viewer, pSpecTcl);

  // first populate
  manager.onGateListChanged();

  auto itemList0 = manager.getItems();
  CPPUNIT_ASSERT( 2 == itemList0.size() );

  // alter the gate list in spectcl
  pList->addCut2D( move(pGate1) );
  pList->removeCut1D( "slice" );

  auto p1DItem = manager.findItem("gate");

  // second populate with same exact set of gates...
  manager.onGateListChanged();

  auto itemList1 = manager.getItems();
  CPPUNIT_ASSERT( 2 == itemList1.size() );

  // first element of list should be the same
  CPPUNIT_ASSERT( p1DItem == manager.findItem("gate") );

  // second element of list should be different
  CPPUNIT_ASSERT( itemList0.at(1) != itemList1.at(1) );
  // Makes sure we have the same items...
}

} // end of namespace
