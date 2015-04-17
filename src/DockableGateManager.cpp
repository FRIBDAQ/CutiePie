//    This software is Copyright by the Board of Trustees of Michigan
//    State University (c) Copyright 2015.
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

static const char* Copyright = "(C) Copyright Michigan State University 2015, All rights reserved";
#include "DockableGateManager.h"
#include "ui_DockableGateManager.h"
#include "GateBuilderDialog.h"
#include "GateBuilder1DDialog.h"
#include "SpectrumViewer.h"
#include "SpecTclInterface.h"
#include "QRootCanvas.h"
#include "GSlice.h"
#include "GGate.h"
#include "GateList.h"

#include "SliceTableItem.h"
#include "GateListItem.h"
#include <QListWidget>
#include <QMessageBox>
#include <TH1.h>
#include <TH2.h>

#include <iostream>

using namespace std;

DockableGateManager::DockableGateManager(const SpectrumViewer& viewer,
                                         SpecTclInterface* pSpecTcl,
                                         QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::DockableGateManager),
    m_view(viewer),
    m_pSpecTcl(pSpecTcl)
{
    ui->setupUi(this);
    connect(ui->addButton, SIGNAL(clicked()), 
            this, SLOT(launchAddGateDialog()));
    connect(ui->editButton, SIGNAL(clicked()), 
            this, SLOT(launchEditGateDialog()));
    connect(ui->deleteButton, SIGNAL(clicked()), 
            this, SLOT(deleteGate()));

    connect(pSpecTcl, SIGNAL(gateListChanged()),
            this, SLOT(onGateListChanged()));
}

DockableGateManager::~DockableGateManager()
{
    delete ui;
}

void DockableGateManager::launchAddGateDialog()
{
    auto pCanvas = m_view.getCurrentFocus();
    auto histPkg = m_view.getCurrentHist();

    // determine whether this is a 1d or 2d hist and 
    // open to appropriate dialog
    if (histPkg->hist()->InheritsFrom(TH2::Class())) {

        GateBuilderDialog* dialog = new GateBuilderDialog(*pCanvas, *histPkg);
        dialog->setAttribute(Qt::WA_DeleteOnClose);

        connect(dialog, SIGNAL(completed(GGate*)), 
                this, SLOT(registerGate(GGate*)));

        dialog->show();
        dialog->raise();
    } else {
        GateBuilder1DDialog* dialog = new GateBuilder1DDialog(*pCanvas, *histPkg);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        connect(dialog, SIGNAL(completed(GSlice*)),
                this, SLOT(registerSlice(GSlice*)));

        dialog->show();
        dialog->raise();
    }
}

void DockableGateManager::launchEditGateDialog()
{
    auto pCanvas = m_view.getCurrentFocus();
    auto histPkg = m_view.getCurrentHist();

    auto selection = ui->gateList->selectedItems();
    if (selection.size()==1) {
        auto pItem = selection.at(0);

        // determine whether this is a 1d or 2d gate and 
        // open to appropriate dialog
        if (auto pSlItem = dynamic_cast<SliceTableItem*>(pItem)) {
            auto pCut = pSlItem->getSlice();
            GateBuilder1DDialog* dialog = new GateBuilder1DDialog(*pCanvas, 
                                                                  *histPkg, pCut);
            dialog->setAttribute(Qt::WA_DeleteOnClose);
            connect(dialog, SIGNAL(completed(GSlice*)),
                    this, SLOT(editSlice(GSlice*)));

            dialog->show();
            dialog->raise();

        } else {
            auto pGateItem = dynamic_cast<GateListItem*>(pItem);
            auto pGate = pGateItem->getGate();

            // make sure that state is updated if user moved the cut via the gui
            pGate->synchronize(GGate::GUI);

            GateBuilderDialog* dialog = new GateBuilderDialog(*pCanvas, *histPkg, pGate);
            dialog->setAttribute(Qt::WA_DeleteOnClose);
            connect(dialog, SIGNAL(completed(GGate*)),
                    this, SLOT(editGate(GGate*)));

            dialog->show();
            dialog->raise();
        }
    } else {
        QMessageBox::warning(0, "Invalid selection", "User must select one gate to edit.");
    }
}

void DockableGateManager::addGateToList(GGate* pCut)
{

    Q_ASSERT(pCut != nullptr);

    GateListItem* pItem = new GateListItem(QString(pCut->getName()),
                                           ui->gateList,
                                           Qt::UserRole,
                                           pCut);

    if (pCut->getType() == SpJs::BandGate) {
      pItem->setIcon(QIcon(":/icons/band-icon.png"));
    } else {
      pItem->setIcon(QIcon(":/icons/contour-icon.png"));
    }
    ui->gateList->addItem(pItem);

    // add the gate to all related histograms
    HistogramList::addGate(pCut);

    auto histPkg = m_view.getCurrentHist();
    if (histPkg) {
      histPkg->draw();
    }
}

void DockableGateManager::registerGate(GGate* pCut)
{
    addGateToList(pCut);

    if (m_pSpecTcl) {
        m_pSpecTcl->addGate(*pCut);
    }

}

void DockableGateManager::addSliceToList(GSlice* pSlice)
{
    Q_ASSERT(pSlice != nullptr);

    QString name = pSlice->getName();

    SliceTableItem* pItem = new SliceTableItem(name,
                                               ui->gateList,
                                               Qt::UserRole,
                                               pSlice);
    pItem->setIcon(QIcon(":/icons/slice-icon.png"));
    ui->gateList->addItem(pItem);

    // add the slice to all related histograms
    HistogramList::addSlice(pSlice);


    auto histPkg = m_view.getCurrentHist();
    if (histPkg) {
        histPkg->draw();
    }
}


void DockableGateManager::registerSlice(GSlice *pSlice)
{
    addSliceToList(pSlice);

    if (m_pSpecTcl) {
        m_pSpecTcl->addGate(*pSlice);
    }
}


void DockableGateManager::editGate(GGate* pCut)
{
    Q_ASSERT( pCut != nullptr );

    if (m_pSpecTcl) {
        m_pSpecTcl->editGate(*pCut);
    }

    auto histPkg = m_view.getCurrentHist();
    if (histPkg) {
      histPkg->draw();
    }
}


void DockableGateManager::editSlice(GSlice *pSlice)
{
    Q_ASSERT(pSlice != nullptr);

    if (m_pSpecTcl) {
        m_pSpecTcl->editGate(*pSlice);
    }

    auto histPkg = m_view.getCurrentHist();
    if (histPkg) {
      histPkg->draw();
    }
}

void DockableGateManager::deleteGate()
{
  auto selected = ui->gateList->selectedItems();
  for ( auto pItem : selected ) {
    // tell SpecTcl to delete the gate
    cout << pItem->text().toStdString() << " @ " << (void*)pItem << endl;
    if (m_pSpecTcl) {
      m_pSpecTcl->deleteGate(pItem->text());
    }
  
    removeGate(pItem);
  }

  auto histPkg = m_view.getCurrentHist();
  if (histPkg) {
    histPkg->draw();
  }

}

void DockableGateManager::clearList()
{
  while ( ui->gateList->count() > 0 ) {
    auto pItem = ui->gateList->item(0);
    cout << (void*) pItem << endl;
    removeGate(pItem);
  }
}

void DockableGateManager::onGateListChanged()
{
  clearList();

  cout << "Update gates!" << endl;

  auto list = m_pSpecTcl->getGateList();

  auto it_1d = list->begin1d();
  auto itend_1d = list->end1d();
  while ( it_1d != itend_1d ) {
      cout << (void*) it_1d->get() << endl;
    HistogramList::addSlice(it_1d->get());
    addSliceToList(it_1d->get());
    ++it_1d;
  }
  
  auto it_2d = list->begin2d();
  auto itend_2d = list->end2d();
  while ( it_2d != itend_2d ) {
      cout << (void*) it_2d->get() << endl;
    HistogramList::addGate(it_2d->get());
    addGateToList(it_2d->get());
    ++it_2d;
  }
}


  
void DockableGateManager::removeGate(QListWidgetItem* pItem) 
{
  // Remove the row
  auto row = ui->gateList->row(pItem);
  ui->gateList->takeItem(row);
  delete pItem;

}
