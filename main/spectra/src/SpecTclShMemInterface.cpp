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

#include "SpecTclShMemInterface.h"
#include "SpecTclRESTInterface.h"
#include "Xamine2Root/HistFiller.h"
#include "dispshare.h"
#include "QRootCanvas.h"
#include "HistogramList.h"
#include "CanvasOps.h"
#include "XamineSpectrumInterface.h"

#include "TCanvas.h"

#include <QMessageBox>

#include <iostream>


namespace Viewer {

SpecTclShMemInterface::SpecTclShMemInterface() :
    SpecTclInterface(),
    m_pRESTInterface(new SpecTclRESTInterface)
{

    
    // we need to shmem key at this point

    Xamine_initspectra(); // sets up access to spectra

    connect(m_pRESTInterface.get(), SIGNAL(histogramContentUpdated(HistogramBundle*)),
            this, SLOT( onHistogramContentUpdated(HistogramBundle*)));

    connect(m_pRESTInterface.get(), SIGNAL(histogramListChanged()),
            this, SLOT(onHistogramListChanged()));

    connect(m_pRESTInterface.get(), SIGNAL(gateListChanged()),
            this, SLOT(onGateListChanged()));

}

SpecTclShMemInterface::~SpecTclShMemInterface()
{
}

void SpecTclShMemInterface::addGate(const GSlice &slice)
{
    m_pRESTInterface->addGate(slice);
}

void SpecTclShMemInterface::editGate(const GSlice &slice)
{
    m_pRESTInterface->editGate(slice);
}

void SpecTclShMemInterface::deleteGate(const GSlice &slice)
{
    m_pRESTInterface->deleteGate(slice);
}

void SpecTclShMemInterface::addGate(const GGate &gate)
{
    m_pRESTInterface->addGate(gate);
}

void SpecTclShMemInterface::editGate(const GGate &gate)
{
    m_pRESTInterface->editGate(gate);
}

void SpecTclShMemInterface::deleteGate(const GGate &gate)
{
    m_pRESTInterface->deleteGate(gate);
}

void SpecTclShMemInterface::deleteGate(const QString &name)
{
    m_pRESTInterface->deleteGate(name);
}

void SpecTclShMemInterface::addOrGate(
    const std::string& name, const std::vector<std::string>& components
)
{
    m_pRESTInterface->addOrGate(name, components);
}
void SpecTclShMemInterface::addAndGate(
    const std::string& name, const std::vector<std::string>& components
) {
    m_pRESTInterface->addAndGate(name, components);
}

void SpecTclShMemInterface::editOrGate(
    const std::string& name, const std::vector<std::string>& components
) {
    m_pRESTInterface->editOrGate(name, components);
}
void SpecTclShMemInterface::editAndGate(
    const std::string& name, const std::vector<std::string>& components
) {
    m_pRESTInterface->editAndGate(name, components);
}


void SpecTclShMemInterface::enableGatePolling(bool enable)
{
    m_pRESTInterface->enableGatePolling(enable);
}

bool SpecTclShMemInterface::gatePollingEnabled() const {
    return m_pRESTInterface->gatePollingEnabled();
}

MasterGateList* SpecTclShMemInterface::getGateList()
{
    return m_pRESTInterface->getGateList();
}

void SpecTclShMemInterface::enableHistogramInfoPolling(bool enable)
{
    m_pRESTInterface->enableHistogramInfoPolling(enable);
}

bool SpecTclShMemInterface::histogramInfoPollingEnabled() const
{
    return m_pRESTInterface->histogramInfoPollingEnabled();
}

HistogramList* SpecTclShMemInterface::getHistogramList()
{
    return m_pRESTInterface->getHistogramList();
}

void SpecTclShMemInterface::requestHistContentUpdate(QRootCanvas *pCanvas)
{
    Q_ASSERT( pCanvas != nullptr );

    auto histNames = CanvasOps::extractAllHistNames(*pCanvas);

    for (auto& name : histNames) {
        // update all histograms in this canvas
        requestHistContentUpdate(name);
    }
}

void SpecTclShMemInterface::requestHistContentUpdate(TVirtualPad *pPad)
{
    Q_ASSERT( pPad != nullptr );

    auto histNames = CanvasOps::extractAllHistNames(*pPad);

    for (auto& name : histNames) {
        // update all histograms in this canvas
        requestHistContentUpdate(name);
    }
}

void SpecTclShMemInterface::requestHistContentUpdate(const QString &hName)
{
    Xamine2Root::HistFiller filler;

    try {
        HistogramBundle* pHBundle = getHistogramList()->getHistFromClone(hName);
        if (pHBundle) {
            filler.fill(pHBundle->getHist(), pHBundle->getName().toStdString());

            // Update the clones
            auto hists = pHBundle->getClones();
            for (auto& histInfo : hists) {
                filler.fill(*(histInfo.second), pHBundle->getName().toStdString());
            }
        }
        emit histogramContentUpdated(pHBundle);
    } catch (std::exception& exc) {
        QMessageBox::warning(nullptr, QString("Histogram Update Error"), QString(exc.what()));
    }
}

/**
 * clearSpectrum
 *    Clear specific spectrum (delegated to the rest interface):
 * @param pCanvas - canvas displaying the spectrum.
 */
void
SpecTclShMemInterface::clearSpectrum(QRootCanvas* pCanvas)
{
    m_pRESTInterface->clearSpectrum(pCanvas);
}
void
SpecTclShMemInterface::clearSpectrum(QString* pName)
{
    m_pRESTInterface->clearSpectrum(pName);
}
/**
 * clearAllSpectra
 *    Clears all spectra in spectcl.
 */
void
SpecTclShMemInterface::clearAllSpectra()
{
    m_pRESTInterface->clearAllSpectra();
}

void SpecTclShMemInterface::onHistogramContentUpdated(HistogramBundle *pBundle) {
    emit histogramContentUpdated(pBundle);
}

void SpecTclShMemInterface::onHistogramListChanged() {
    emit histogramListChanged();
}

void SpecTclShMemInterface::onGateListChanged() {
    emit gateListChanged();
}

} // namespace VIewer
